import os
import argparse
import numpy as np
import time
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader, Subset
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torchvision import datasets, transforms
from torch_ema import ExponentialMovingAverage

from EBMs import create_ebm
from ddpm_unet import create_net
from init import *
from utils_train import *

def _map_train_Diff(index, args):

    device = xm.xla_device()
    # Print the number of available devices
    num_devices = xm.xrt_world_size()
    xm.master_print(f"Number of devices: {num_devices}")
    
    # Create gen (unet)
    gen = create_net(args, args.net_type, args.im_sz, args.net_nf)
    # gen.load_state_dict(torch.load( '/home/alice/models/diffusions/cincic10_imagenet_subset/2024_02_19_17_20/gen_epoch_260.pt', map_location=torch.device('cpu')))
    gen.to(device)
    # gen = WRAPPED_MODEL.to(device)
    if args.ema:
        ema = ExponentialMovingAverage(gen.parameters(), decay=0.9999)
    else:
        ema = None

    diffusion = init_diff(args)
    t_sampler = UniformSampler(diffusion)

    if args.ebm is not None:
        # Create EBM
        ebm = create_ebm(args.ebm,args.image_dims,args.num_filters)
        ebm.load_state_dict(torch.load('/home/alice/data/models/ebms/ebm_cinic10_imagenet.pt', map_location=torch.device('cpu')))
        ebm.to(device)
        ebm.eval()
    else:
        ebm = None
    # Create optimizer
    optim = torch.optim.AdamW(gen.parameters(), lr=args.net_lr)
    # optim.load_state_dict(torch.load('/home/alice/models/diffusions/cincic10_imagenet_subset/2024_02_19_17_20/optim_epoch_260.pt', map_location=torch.device('cpu')))


    #######################
    # ## Data Loading  # ##
    #######################

    if args.poison_Narcissus:

        train_data, poison_indices_all = get_train_data(args.dataset, args.data_dir,False,False,args.poison_Narcissus,args.poison_amount)
        if args.verbose: xm.master_print(f'Len poison indices: {len(poison_indices_all)}, num unique: {len(np.unique(poison_indices_all))}')
        # Save poison indices
        if xm.is_master_ordinal():
            torch.save(poison_indices_all,os.path.join(args.output_dir,f'poison_indices.pt'))
    
    else:
        train_data = get_train_data(args.dataset, args.data_dir,False,False)

    test_data = get_test_data(args.dataset, args.data_dir)

    if args.verbose: xm.master_print(f'Dataset {args.dataset} loaded with {len(train_data)} images.')
 
    if args.verbose: xm.master_print('Setting up data loaders...')
  # Creates a sampler for distributing the data across all TPU cores for training, and a separate sampler for the persistent bank, and a separate sampler for the FID calculation
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(), shuffle=True)
    bank_sampler = torch.utils.data.distributed.DistributedSampler(train_data,num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(),shuffle=True)
    fid_train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(),shuffle=True)
    fid_test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(),shuffle=True)

    # Creates dataloaders, which load data in batches
    train_loader = DataLoader(train_data,batch_size=args.batch_size,sampler=train_sampler,num_workers=args.num_workers,drop_last=True)
    bank_loader = DataLoader(train_data,batch_size=args.batch_size,sampler=bank_sampler,num_workers=args.num_workers,drop_last=True)
    fid_train_loader = DataLoader(train_data,batch_size=args.batch_size,sampler=fid_train_sampler,num_workers=args.num_workers,drop_last=True)
    fid_test_loader = DataLoader(test_data,batch_size=args.batch_size,sampler=fid_test_sampler,num_workers=args.num_workers,drop_last=True)

   
    # banks where langevin samples have been updated many times and are far from data samples
    image_bank_update = initialize_persistent(args.image_dims, args.persistent_size, bank_loader, args.data_epsilon, \
    device, poisoned = args.poison_Narcissus, persistent_path=args.bank_path)
    # image_bank_update = xm.all_gather(image_bank_update, 0)
    image_bank_fixed = image_bank_update.clone()
    image_bank_update += args.data_epsilon_init * torch.randn_like(image_bank_update)
    # banks where langevin samples are close to data samples
    image_bank_burnin = initialize_persistent(args.image_dims, args.persistent_size, bank_loader, args.data_epsilon, \
    device, poisoned = args.poison_Narcissus, persistent_path=args.bank_burin_path)
    # image_bank_burnin = xm.all_gather(image_bank_burnin, 0)
    image_bank_fixed_burnin = image_bank_burnin.clone()
    image_bank_burnin += args.data_epsilon_init * torch.randn_like(image_bank_burnin)
    # counts for how many times burnin images have been updated
    burnin_counts = torch.randint(args.threshold, size=[image_bank_burnin.shape[0]], device=device)
    if args.verbose: xm.master_print(f'Persistent image bank initialized with {image_bank.shape[0]} images.')

    # Print num poison images
    if args.poison_Narcissus:
        sum = 0
        for batch in train_loader:
            (X_batch, y_batch, _, p) = batch
            sum += torch.sum(p).item()

        xm.master_print(f'Number of poison images: {xm.mesh_reduce("sum",sum,np.sum)}')

    #######################
    # ## LEARNING LOOP # ##
    #######################

    xm.master_print('Training has begun.')
    train_losses = []

    # Initialize FID scores dictionary
    fid_train_scores = {}
    fid_test_scores = {}
    for purify_steps in args.fid_purify_steps:
        fid_train_scores[purify_steps] = {}
        fid_test_scores[purify_steps] = {}

    # Initialize tqdm on master device only
    if xm.is_master_ordinal():
        pbar = tqdm(total=args.epochs,ncols=100)

    for epoch in range(1,args.epochs+1):
        train_sampler.set_epoch(epoch)
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        para_fid_train_loader = pl.ParallelLoader(fid_train_loader, [device]).per_device_loader(device)
        para_fid_test_loader = pl.ParallelLoader(fid_test_loader, [device]).per_device_loader(device)

        train_loss_epoch = 0

        for batch_num, batch in enumerate(para_train_loader):

            if args.poison_Narcissus:
                (X_batch, y_batch, _, _) = batch
            else:
                (X_batch, y_batch) = batch

            X_train = X_batch.to(device)

            # obtain samples from ebm for mature states
            images_init, images_target, rand_inds = \
                initialize_fixer_update(args, image_bank_update, image_bank_fixed)
            images_samp, grad_norm = sample_ebm(ebm, images_init, args.mcmc_steps, args.mcmc_temp, args.epsilon)
            # obtain samples from ebm for burnin states
            images_init_burnin, images_target_burnin, rand_inds_burnin = \
                initialize_fixer_update(args, image_bank_burnin, image_bank_fixed_burnin)
            burnin_counts_batch = burnin_counts[rand_inds_burnin]
            images_samp_burnin, _ = sample_ebm(ebm, images_init_burnin, args.mcmc_steps, args.mcmc_temp, args.epsilon)
            
            loss_denoise = \
                update_gen_ddpm(args, gen, diffusion, t_sampler, optim, images_target, images_samp, ema)
            
             # update persistent banks
            image_bank_update, image_bank_fixed, image_bank_burnin, image_bank_fixed_burnin, burnin_counts = \
                update_persistent_fixer_2(args, X_batch, image_bank_update, image_bank_fixed, 
                                          image_bank_burnin, image_bank_fixed_burnin, 
                                          images_samp, images_target, rand_inds, 
                                          images_samp_burnin, images_target_burnin, rand_inds_burnin, 
                                          burnin_counts, burnin_counts_batch)

            # Update Loss and Grad Norm
            train_loss_epoch += loss_denoise.item()

            # Set description on master device only
            if xm.is_master_ordinal():
                pbar.set_description(f'Epoch {epoch}/{args.epochs} Iter {batch_num+1}/{len(para_train_loader)} | Loss: {train_loss_epoch/(batch_num+1):.3e}')

        # Update tqdm on master device only
        if xm.is_master_ordinal():
            pbar.update(1)


        # Update training record for all cores
        train_loss_epoch /= (batch_num + 1)
        train_losses.append(xm.mesh_reduce('train_losses',round(float(train_loss_epoch), 4),np.mean))

        # Save checkpoints and final model
        if (epoch % args.checkpoint_freq == 0 or epoch in [1, args.epochs]):
            # Save model
            xm.save(gen.state_dict(), os.path.join(args.output_dir,f'gen_epoch_{epoch}.pt'))
            # Save optimizer
            xm.save(optim.state_dict(), os.path.join(args.output_dir,f'optim_epoch_{epoch}.pt'))

        if (epoch % args.checkpoint_freq == 0 or epoch in [1,args.epochs]):
            # Save train plots and sample images
            if xm.is_master_ordinal():
                if args.ebm is not None:
                    plot_checkpoint(train_losses, grad_norm=None, image_samples=images_samp.detach().cpu().numpy(), epoch=epoch,\
                     save_path=os.path.join(args.output_dir,f'ebm_epoch_{epoch}.png'))
       
                plot_checkpoint_gen(args, train_losses, images_target, image_samples=images_samp, gen=gen, diffusion=diffusion, \
                 epoch=epoch, save_path=os.path.join(args.output_dir,f'gen_epoch_{epoch}.png'))

            # FID score
            if args.verbose: xm.master_print(f'Calculating FID scores at t-steps {args.fid_purify_steps}...')
            for purify_steps in args.fid_purify_steps:
                fid_train_scores[purify_steps][epoch] = fid_score_calculation(para_fid_train_loader, device, epoch, args, \
                os.path.join(args.output_dir,f'fid_train_epoch_{epoch}_steps_{purify_steps}.png'), mcmc_steps=args.mcmc_steps, purify_steps=purify_steps, \
                poisoned=args.poison_Narcissus, ebm=ebm, gen=gen, diffusion=diffusion)
                
                fid_test_scores[purify_steps][epoch] = fid_score_calculation(para_fid_test_loader, device, epoch, args, \
                os.path.join(args.output_dir,f'fid_test_epoch_{epoch}_steps_{purify_steps}.png'), mcmc_steps=args.mcmc_steps, purify_steps=purify_steps, \
                poisoned=args.poison_Narcissus, ebm=ebm, gen=gen, diffusion=diffusion)

            # Save FID and Losses in once dictionary
            if xm.is_master_ordinal():
                torch.save({'train_losses': train_losses, 'fid_train_scores': fid_train_scores, 'fid_test_scores': fid_test_scores}, os.path.join(args.output_dir,f'training_record_epoch_{epoch}.pt'))

            
if __name__ == '__main__':

    os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'

    parser = argparse.ArgumentParser(description='Diffusion Training')

    # EBM Type and Dataset
    parser.add_argument('--dataset', type=str, default='cincic10_imagenet_subset', metavar='N',choices=['cifar10','cinic10','cifar10_BP','cifar10_GM','cifar10_45K','cinic10','mnist','cincic10_imagenet_subset'], help='Dataset to train on')
    parser.add_argument('--image_dims', default=[3,32,32], nargs='+', type=int, metavar='N')
    parser.add_argument('--ebm', type=str, default='EBMSNGAN32', metavar='N',choices=['EBM_Small', 'EBMSNGAN32', 'EBMSNGAN128', 'EBMSNGAN256'])
    parser.add_argument('--num_filters', type=int, default=128, metavar='N')
    parser.add_argument('--seed', type=int, default=11, metavar='N')

    # Data poisoning
    parser.add_argument('--poison_Narcissus', action='store_true',default=False, help='Whether to poison the Narcissus dataset')
    parser.add_argument('--poison_amount', type=int, default=500, metavar='N', help='Number of images to poison (per call, 5k max)')

    # Langevin dynamics and EBM sampling
    parser.add_argument('--mcmc_steps', type=int, default=100, metavar='N')
    parser.add_argument('--epsilon', type=float, default=1.25e-2, metavar='N')
    parser.add_argument('--mcmc_temp', type=float, default=1e-4, metavar='N')
    parser.add_argument('--data_epsilon', type=float, default=1.5e-2, metavar='N')
    parser.add_argument('--data_epsilon_init', type=float, default=1.5e-2, metavar='N')
    parser.add_argument('--persistent_size', type=int, default=1250, metavar='N', help='persistent bank size on each device')
    parser.add_argument('--threshold', type=int, default=1, metavar='N')
    parser.add_argument('--bank_path', default=None, type=str, help='persistent bank weight path')
    parser.add_argument('--bank_burin_path', default=None, type=str, help='persistent bank burin weight path ')
    

    # Diffusion model type and training
    parser.add_argument('--diff_output', default='epsilon', type=str, choices=['epsilon','start_x'],  help='diffusion model output')
    parser.add_argument('--net_type', default='ddpm_unet_fixer', type=str, choices=['ddpm_unet','ddpm_unet_fixer'], help='unet type')
    parser.add_argument('--t_schedule', default='cosine', type=str, choices=['linear','cosine'], help='t schedule')
    parser.add_argument('--ema', default=True, type=bool, help='EMA option')
    parser.add_argument('--num_t_steps', default=1000, type=int,  help='training t-steps for diffuion model')
    parser.add_argument('--im_sz', default=32, type=int,   help='image size')
    parser.add_argument('--net_nf', default=128, type=int,  help='number of filters for the unet model')
    parser.add_argument('--purify_t', default=125, type=int,  help='number of purify t-steps for the unconditional diffuion model')

    # General training
    parser.add_argument('--epochs', type=int, default=780, metavar='N')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N')
    parser.add_argument('--net_lr', type=float, default=1e-4, metavar='N')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N')
    parser.add_argument('--checkpoint_freq', type=int, default=65, metavar='N', help='Number of checkpoints to save during training')
    parser.add_argument('--fid_purify_steps', nargs='+', type=int, default=[100, 125], help='List of t steps to calculate FID at')

    # Other training options
    parser.add_argument('--data_dir', type=str, default='/home/alice/data', metavar='N')
    parser.add_argument('--output_dir', type=str, default='/home/alice/models/bank_diffusions', metavar='N')
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

    # Create Output Directory (add timestamp)
    if not args.poison_Narcissus:
        args.output_dir = os.path.join(args.output_dir,f'{args.dataset}',time.strftime("%Y_%m_%d_%H_%M", time.localtime()))
    else:
        args.output_dir = os.path.join(args.output_dir,f'{args.dataset}_Narcissus_{args.poison_amount}',time.strftime("%Y_%m_%d_%H_%M", time.localtime()))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Print args into txt file in output directory
    with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

    # #create gen (unet), using WRAPPED_MODEL causes device error
    # gen = create_net(args, args.net_type, args.im_sz, args.net_nf)
    # WRAPPED_MODEL = xmp.MpModelWrapper(gen)

    if args.verbose:
        print(f'Using {args.net_type} on {args.dataset}, Number of parameters: {sum(p.numel() for p in ebm.parameters())}')

    start_time = time.time()
    xmp.spawn(_map_train_Diff, args=(args,), nprocs=8, start_method='fork')
    print(f'Training Complete! Time taken: {time.time() - start_time} seconds')