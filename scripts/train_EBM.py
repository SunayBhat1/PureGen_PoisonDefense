import os
import argparse
import numpy as np
import time
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torchvision import datasets, transforms
from diffusers.optimization import get_cosine_schedule_with_warmup

# Add parent directory to sys path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.EBMs import create_ebm
from utils.ebm_train import *
from utils.gen_model_data import get_train_data, get_test_data, dataset_dict

def _map_train_EBM(index,args, WRAPPED_MODEL):

    device = xm.xla_device()

    # Create EBM
    ebm = WRAPPED_MODEL.to(device)

    #######################
    # ## Data Loading  # ##
    #######################

    if args.poison_Narcissus:

        train_data, poison_indices_all = get_train_data(args.dataset, args.data_dir,args.use_rand_trans,args.img_resize,args.poison_Narcissus,args.poison_amount,args.noise_sz_narcissus,args.noise_eps_narcissus)
        if args.verbose: xm.master_print(f'Len poison indices: {len(poison_indices_all)}, num unique: {len(np.unique(poison_indices_all))}')
        # Save poison indices
        if xm.is_master_ordinal():
            torch.save(poison_indices_all,os.path.join(args.output_dir,f'poison_indices.pt'))
    else:

        train_data = get_train_data(args.dataset, args.data_dir,args.use_rand_trans,args.img_resize)

    test_data = get_test_data(args.dataset, args.data_dir,args.img_resize)

    if args.verbose: xm.master_print(f'Dataset {args.dataset} loaded with {len(train_data)} images.')
 
    if args.verbose: xm.master_print('Setting up data loaders...')
    # Creates a sampler for distributing the data across all TPU cores for training, and a separate sampler for the persistent bank, and a separate sampler for the FID calculation
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(), shuffle=True)
    bank_sampler = torch.utils.data.distributed.DistributedSampler(train_data,num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(),shuffle=True)
    fid_sampler = torch.utils.data.distributed.DistributedSampler(test_data,num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(),shuffle=True)

    # Creates dataloaders, which load data in batches
    train_loader = DataLoader(train_data,batch_size=args.batch_size,sampler=train_sampler,drop_last=True)
    bank_loader = DataLoader(train_data,batch_size=args.batch_size,sampler=bank_sampler,drop_last=True)
    fid_loader = DataLoader(test_data,batch_size=16,sampler=fid_sampler,drop_last=True)

    # Create optimizer
    optim = torch.optim.Adam(ebm.parameters(), lr=args.lr)

    # Create learning rate scheduler
    if args.lr_schedule == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=args.lr_decay_milestones, gamma=args.lr_decay_factor)
    else:
        # Cosine Annealing with warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=optim,
                    num_warmup_steps=args.lr_warmup,
                    num_training_steps=(len(train_loader) * args.epochs),
                )

    # Create persistent image bank
    if args.img_resize is None:
        args.image_dims = [3,dataset_dict[args.dataset]['image_dim'],dataset_dict[args.dataset]['image_dim']]
    else:
        args.image_dims =[3,args.img_resize,args.img_resize]
    image_bank = initialize_persistent(args.image_dims, args.persistent_size, bank_loader, args.data_epsilon, device, poisoned = args.poison_Narcissus)
    if args.verbose: xm.master_print(f'Persistent image bank initialized with {image_bank.shape[0]} images.')

    # Print num poison images
    if args.poison_Narcissus:
        sum = 0
        for batch in train_loader:
            (_, _, _, p) = batch
            sum += torch.sum(p).item()

        xm.master_print(f'Number of poison images: {xm.mesh_reduce("sum",sum,np.sum)}')

    #######################
    # ## LEARNING LOOP # ##
    #######################

    xm.master_print('Training has begun.')
    ebm_losses = []
    grad_norms = []
    learning_rates = []

    # Initialize FID scores dictionary
    fid_scores = {}
    for mcmc_steps in args.fid_mcmc_steps:
        fid_scores[mcmc_steps] = {}

    # Initialize tqdm on master device only
    if xm.is_master_ordinal():
        pbar = tqdm(total=args.epochs,ncols=100)

    for epoch in range(1,args.epochs+1):
        train_sampler.set_epoch(epoch)
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        para_fid_loader = pl.ParallelLoader(fid_loader, [device]).per_device_loader(device)

        ebm_loss_epoch = 0
        gard_norms_epoch = 0

        for batch_num, batch in enumerate(para_train_loader):

            if args.poison_Narcissus:
                (X_batch, _, _, _) = batch
            elif args.dataset == 'stl10':
                X_batch = batch[0]
            else:
                (X_batch, _) = batch

            # obtain positive samples from data
            X_train = X_batch.to(device)
            images_data = sample_data(X_train, args.data_epsilon)
            
            # obtain negative samples from model
            images_init, rand_inds = initialize_mcmc(args.batch_size, image_bank=image_bank)
            images_samp, grad_norm = sample_ebm(ebm, images_init, args.mcmc_steps, args.mcmc_temp, args.epsilon)

            # load persistent states back into model
            image_bank = update_persistent(images_data, image_bank, images_samp, rand_inds, args.data_epsilon, args.rejuv_prob)
            loss = (ebm(images_data).mean() - ebm(images_samp).mean()) / args.mcmc_temp

            # update ebm weights
            optim.zero_grad()
            loss.backward()
            xm.optimizer_step(optim)
            
            # Update Loss and Grad Norm
            ebm_loss_epoch += loss.item()
            gard_norms_epoch += grad_norm.item()

            if args.lr_schedule == 'cosine':
                learning_rates.append(optim.param_groups[0]['lr'])
                lr_scheduler.step()

            # Set description on master device only
            if xm.is_master_ordinal():
                pbar.set_description(f'Epoch {epoch}/{args.epochs} Iter {batch_num+1}/{len(para_train_loader)} | Loss: {ebm_loss_epoch/(batch_num+1):.3e}| Grad Norm: {gard_norms_epoch/(batch_num+1):.3e}')

        # Update tqdm on master device only
        if xm.is_master_ordinal():
            learning_rates.append(optim.param_groups[0]['lr'])
            pbar.update(1)

        # Update learning rate
        if args.lr_schedule == 'multistep':
            lr_scheduler.step()

        # Update training record for all cores
        ebm_loss_epoch /= (batch_num + 1)
        gard_norms_epoch /= (batch_num + 1)
        ebm_losses.append(xm.mesh_reduce('ebm_losses',round(float(ebm_loss_epoch), 4),np.mean))
        grad_norms.append(xm.mesh_reduce('gard_norms',round(float(gard_norms_epoch), 4),np.mean))

        # Save checkpoints and final model
        if (epoch % args.checkpoint_freq == 0 or epoch in [1,5,10,args.epochs]):
            
            # Save model
            xm.save(ebm.state_dict(), os.path.join(args.output_dir,f'ebm_epoch_{epoch}.pt'))

            # Save train plots and sample images
            if xm.is_master_ordinal():
                plot_checkpoint(ebm_losses, grad_norms, images_samp.detach().cpu().numpy(), epoch, os.path.join(args.output_dir,f'epoch_{epoch}.png'))

            # FID score
            if args.verbose: xm.master_print(f'Calculating FID scores at steps {args.fid_mcmc_steps}...')
            for mcmc_steps in args.fid_mcmc_steps:
                fid_scores[mcmc_steps][epoch] = fid_score_calculation(ebm, para_fid_loader, device, epoch, args, mcmc_steps, os.path.join(args.output_dir,f'fid_epoch_{epoch}_steps_{mcmc_steps}.png'),batch_num_break=dataset_dict[args.dataset]['fid_batch_num'])
                xm.rendezvous('fid_scores')
            # Save FID and Losses in once dictionary
            if xm.is_master_ordinal():
                torch.save({'ebm_losses': ebm_losses, 'grad_norms': grad_norms, 'learning_rates':learning_rates, 'fid_scores': fid_scores}, os.path.join(args.output_dir,f'logs.pt'))
            
if __name__ == '__main__':

    os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'

    parser = argparse.ArgumentParser(description='EBM Training')

    parser.add_argument('--remote_user', type=str, help='username for the remote server (TPU only, else pass in full directory args below)')

    # Model
    parser.add_argument('--model', type=str, default='EBMSNGAN32', metavar='N',choices=['EBM', 'EBMSNGAN32', 'EBMSNGAN128'], help='Model to train')
    parser.add_argument('--num_filters', type=int, default=128, metavar='N')
    parser.add_argument('--seed', type=int, default=11, metavar='N')

    # Data
    parser.add_argument('--dataset', type=str, default='cincic10_imagenet', metavar='N',choices=['cifar10','cinic10','cincic10_imagenet','tiny-imagenet-200','textures','office_home','stl10','caltech256','flowers102','lfw_people','food101','fgvc_aircraft','oxford_iiit_pet'], help='Dataset to train on')
    parser.add_argument('--img_resize', type=int, default=None, metavar='N', help='Image resize dimension')
    parser.add_argument('--fid_mcmc_steps', nargs='+', type=int, default=[500], help='List of mcmc steps to calculate FID at')


    # Langevin dynamics and EBM training
    parser.add_argument('--mcmc_steps', type=int, default=100, metavar='N')
    parser.add_argument('--epsilon', type=float, default=1.25e-2, metavar='N')
    parser.add_argument('--mcmc_temp', type=float, default=1e-4, metavar='N')
    parser.add_argument('--data_epsilon', type=float, default=1.5e-2, metavar='N')
    parser.add_argument('--persistent_size', type=int, default=1250, metavar='N', help='persistent bank size on each device')
    parser.add_argument('--rejuv_prob', type=float, default=0.05, metavar='N', help='probability of rejuvenating persistent samples')

    # General training and optimizer
    parser.add_argument('--epochs', type=int, default=120, metavar='N')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='N')
    parser.add_argument('--lr_schedule', type=str, default='multistep', metavar='N',choices=['cosine','multistep'], help='Learning rate schedule')
    parser.add_argument('--lr_warmup', type=int, default=500, metavar='N', help='Number of warmup steps for cosine schedule')
    parser.add_argument('--lr_decay_milestones', nargs='+', type=int, default=[40, 75, 100], help='List of epoch indices to decrease learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Factor by which learning rate is decreased at each milestone')
    parser.add_argument('--use_rand_trans', action='store_true',default=False, help='Whether to use random transformations for data augmentation')
    parser.add_argument('--checkpoint_freq', type=int, default=20, metavar='N', help='Number of checkpoints to save during training')

    # Data poisoning
    parser.add_argument('--poison_Narcissus', action='store_true',default=False, help='Whether to poison the Narcissus dataset')
    parser.add_argument('--poison_amount', type=int, default=5000, metavar='N', help='Number of images to poison (per class, 5k max)')
    parser.add_argument('--noise_sz_narcissus', default=32, type=int, help='size of the noise trigger for Narcissus')
    parser.add_argument('--noise_eps_narcissus', default=8, type=int, help='epsilon for the noise trigger for Narcissus')

    # Other training options
    parser.add_argument('--data_dir', type=str, default='/home/data', metavar='N')
    parser.add_argument('--output_dir', type=str, default='/home/models', metavar='N')
    parser.add_argument('--verbose','--v', action='store_true')

    args = parser.parse_args()

    # Raise erros
    if args.dataset == 'cinic10' and args.poison_Narcissus and args.poison_amount > 2000:
        raise ValueError('Cinic10 dataset has only 2000 cifar images per class. Please choose a smaller poison amount.')

    # Print Arguments
    if args.verbose:
        print('\n_____Arguments_____\n')
        for arg in vars(args):
            print(f'{arg}: {getattr(args, arg)}')
        print('___________________\n')

    # Setup directories for remote server
    if args.remote_user is not None:
        args.data_dir = args.data_dir.replace('/home',f'/home/{args.remote_user}')
        args.output_dir = args.output_dir.replace('/home',f'/home/{args.remote_user}')

    # Create Output Directory (add timestamp)
    if not args.poison_Narcissus:
        args.output_dir = os.path.join(args.output_dir,f'{args.model}',f'{args.dataset}',f'nf[{args.num_filters}]_' + time.strftime("%Y_%m_%d_%H_%M", time.localtime()))
    else:
        args.output_dir = os.path.join(args.output_dir,f'{args.model}',f'{args.dataset}_NS[num={args.poison_amount}_size={args.noise_sz_narcissus}_eps={args.noise_eps_narcissus}]',f'nf[{args.num_filters}]_' + time.strftime("%Y_%m_%d_%H_%M", time.localtime()))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Print args into txt file in output directory
    with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

    # Create EBM
    ebm = create_ebm(args.model,num_filters=args.num_filters)
    WRAPPED_MODEL = xmp.MpModelWrapper(ebm)

    if args.verbose:
        print(f'Using {args.model} on {args.dataset}, Number of parameters: {sum(p.numel() for p in ebm.parameters())}')

    start_time = time.time()
    xmp.spawn(_map_train_EBM, args=(args, WRAPPED_MODEL), nprocs=8, start_method='fork')
    print(f'Training Complete! Time taken: {time.time() - start_time} seconds')