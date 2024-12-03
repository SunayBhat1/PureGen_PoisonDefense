import torch
from torch.utils.data import DataLoader
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from tqdm import tqdm
import os
import argparse
import time
import numpy as np

from diffusers import DDPMScheduler

# Add parent directory to sys path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.DDPMs import create_diff
from PureGen.utils.ddpm_train import *
from utils.gen_model_data import get_train_data, get_test_data


def _map_train(index,args):
    device = xm.xla_device()

    diff_model = create_diff(args.model, args.unet_channels, args.nf, args.num_res_blocks)
    diff_model = diff_model.to(device)
    
    if args.verbose: xm.master_print(f'Training Diffussion {args.model} Model with num params: {sum(p.numel() for p in diff_model.parameters())}')


    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Get train and test data
    if args.poison_Narcissus:

        train_data, poison_indices_all = get_train_data(args.dataset, args.data_dir,args.use_rand_trans,args.img_resize,args.poison_Narcissus,args.poison_amount,args.noise_sz_narcissus,args.noise_eps_narcissus)
        if args.verbose: xm.master_print(f'Len poison indices: {len(poison_indices_all)}, num unique: {len(np.unique(poison_indices_all))}')
        # Save poison indices
        if xm.is_master_ordinal():
            torch.save(poison_indices_all,os.path.join(args.output_dir,f'poison_indices.pt'))
    else:
        train_data = get_train_data(args.dataset, args.data_dir,args.use_rand_trans,args.img_resize)
    test_data = get_test_data(args.dataset, args.data_dir,args.img_resize)

    if args.verbose: 
        xm.master_print(f'Dataset {args.dataset} loaded with {len(train_data)} images.')
        xm.master_print(f'Test Dataset {args.dataset} loaded with {len(test_data)} images.')

    # Creates a sampler for distributing the data across all TPU cores for tdiff_modelraining, and a separate sampler for the persistent bank, and a separate sampler for the FID calculation
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(), shuffle=True)

    # Creates dataloaders, which load data in batches
    train_loader = DataLoader(train_data,batch_size=args.batch_size,sampler=train_sampler,drop_last=True)
    test_loader = DataLoader(test_data,batch_size=512)

    optim, lr_scheduler = get_optimizer_scheduler(diff_model, args.optimizer, args.lr, 
                                                    lr_schedule=args.lr_schedule, 
                                                    train_iters=len(train_loader) * args.epochs,
                                                    lr_warmup=args.lr_warmup,
                                                    lr_milestones=args.lr_decay_milestones,
                                                )


        
    if args.verbose:
        xm.master_print(f'Optimizer: {args.optimizer} - LR: {args.lr} - LR Schdule: {args.lr_schedule}')

    xm.master_print('Training has begun.')

    criterion = torch.nn.MSELoss()

    diff_losses = []
    learning_rates = []

    # Initialize tqdm on master device only
    if xm.is_master_ordinal():
        pbar = tqdm(total=args.epochs,ncols=100)

    # Epoch Loops
    for epoch in range(1,args.epochs+1):
        train_sampler.set_epoch(epoch)
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)

        diff_model.train()
 
        # Logging variables
        diff_loss = 0

        # Train loop
        for batch_num, batch in enumerate(para_train_loader):

            if args.poison_Narcissus:
                (X_train, _, _, _) = batch
            else:
                (X_train, _) = batch

            

            ### Sample noise and timesteps
            noise = torch.randn(X_train.shape, device=X_train.device)
            timesteps = torch.randint(0, args.ddpm_steps, (X_train.shape[0],), device=X_train.device,dtype=torch.int64)

            ### Add noise to images
            noisy_images = noise_scheduler.add_noise(X_train, noise, timesteps)

            inputs = noisy_images
            t_s = timesteps

            target = noise

            noise_predict = diff_model(inputs, timestep=t_s,return_dict=False)[0]

            # Calculate loss and backprop
            loss = criterion(noise_predict, target)

            optim.zero_grad()
            loss.backward()
            xm.optimizer_step(optim)

            # Update learning rate (Cosine)
            if args.lr_schedule == 'cosine':
                learning_rates.append(optim.param_groups[0]['lr'])
                lr_scheduler.step()

            diff_loss += loss.item()

            # Set description on master device only
            if xm.is_master_ordinal():
                pbar.set_description(f'Epoch [{epoch}/{args.epochs}] Iter [{batch_num+1}/{len(para_train_loader)}] | Loss: {loss.item():.4e}')

        # Update learning rate (Multi Step)
        if args.lr_schedule == 'multi_step':
            learning_rates.append(optim.param_groups[0]['lr'])
            lr_scheduler.step()

        # Update progress bar on master device only
        if xm.is_master_ordinal():
            pbar.update(1)

        # Aggregate loss across all TPU cores
        diff_loss /= len(para_train_loader)
        diff_losses.append(xm.mesh_reduce('loss_reduce', diff_loss, np.mean))

        xm.mark_step()

        # Save checkpoints and final model
        if (epoch % args.checkpoint_freq == 0 or epoch in [1,5,10,args.epochs]):

            if args.verbose:
                xm.master_print(f'Saving Checkpoint at Epoch {epoch}')

            # Save model
            xm.save(diff_model.state_dict(), os.path.join(args.output_dir,f'diff_model_{epoch}.pt'))

            if xm.is_master_ordinal():

                noise_scheduler.save_pretrained(os.path.join(args.output_dir,f'noise_scheduler.pt'))

                # Save Losses and Learning Rates
                torch.save({'diff_losses': diff_losses, 'learning_rates': learning_rates}, os.path.join(args.output_dir,f'logs.pt'))

                # Collect test images
                X_test = next(iter(test_loader))[0]
                                
                # Reverse process Diffusions model
                diff_model.eval()

                MCMC_STEPS = 500

                DIFF_STEPS = 75
                with torch.no_grad():
                    noise = torch.randn(X_test.shape).to(device)
                    X_test_fixed = noise_scheduler.add_noise(X_test,noise,timesteps = torch.LongTensor([DIFF_STEPS]))
                    
                    for  t in reversed(range(DIFF_STEPS)):
                        # 1. predict noise residual
                        residual = diff_model(X_test_fixed, t).sample

                        # 2. compute previous image and set x_t -> x_t-1
                        X_test_fixed = noise_scheduler.step(residual, t, X_test_fixed).prev_sample

                        xm.mark_step()
                        
                try: fid_score = fid_score_calculation(X_test,X_test_fixed,device)
                except: fid_score = -1

                plot_checkpoint(diff_losses,X_test[0:16].cpu(),None,X_test_fixed[0:16].cpu(),epoch,os.path.join(args.output_dir,f'epoch_{epoch}.pdf'),
                                    diff_fid=fid_score,
                                    diff_steps=DIFF_STEPS)

    # Renedevous training end
    xm.rendezvous('training end!')
    return


if __name__ == '__main__':

    os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'

    parser = argparse.ArgumentParser(description='Diffusion Fixer Training')

    parser.add_argument('--remote_user', type=str, help='username for the remote server (TPU only, else pass in full directory args below)')
    parser.add_argument('--seed', type=int, default=11, metavar='N')

    # Diffuion Model
    parser.add_argument('--model', type=str, default='DM_UNET', metavar='N',choices=['DM_UNET', 'DM_UNET_S', 'DM_CONV','DM_UNET_DUBHEAD'], help='Model to train')
    parser.add_argument('--unet_channels', type=int, default=(128, 128, 256, 256, 512, 512), nargs='+', help='Number of channels in each block of the UNet')
    parser.add_argument('--nf', type=int, default=64, help='Number of features')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks')

    # Data
    parser.add_argument('--dataset', type=str, default='cincic10_imagenet', metavar='N',choices=['cifar10','cinic10','cincic10_imagenet','tiny-imagenet-200','office_home','textures','stl10','caltech256','flowers102','lfw_people','food101','fgvc_aircraft','oxford_iiit_pet'], help='Dataset to train on')
    parser.add_argument('--img_resize', type=int, default=None, metavar='N', help='Image resize dimension')
    parser.add_argument('--bank_size', type=int, default=320, metavar='N', help='Size of the persistent bank')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N')
    parser.add_argument('--ddpm_steps', type=int, default=250, metavar='N', help='Number of steps for DDPM training')
    parser.add_argument('--use_rand_trans',default=False,action='store_true',help='Use random transformations on the dataset')
    parser.add_argument('--poison_Narcissus', action='store_true',default=False, help='Whether to poison the Narcissus dataset')
    parser.add_argument('--poison_amount', type=int, default=5000, metavar='N', help='Number of images to poison (per class, 5k max)')
    parser.add_argument('--noise_sz_narcissus', default=32, type=int, help='size of the noise trigger for Narcissus')
    parser.add_argument('--noise_eps_narcissus', default=8, type=int, help='epsilon for the noise trigger for Narcissus')

    # General training and optimizer
    parser.add_argument('--epochs', type=int, default=150, metavar='N')
    parser.add_argument('--optimizer', type=str, default='adamw', metavar='N',choices=['adam','sgd'], help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='N', help='Learning rate')
    parser.add_argument('--lr_schedule', type=str, default='cosine', metavar='N',choices=['cosine','multi_step'], help='Learning rate scheduler')
    parser.add_argument('--lr_warmup', type=int, default=500, metavar='N', help='Number of warmup steps')
    parser.add_argument('--lr_decay_milestones', nargs='+', type=int, default=[0,20,40], help='List of epoch indices to decrease learning rate')
    parser.add_argument('--checkpoint_freq', type=int, default=10, metavar='N', help='Number of checkpoints to save during training')

    # Other training options
    parser.add_argument('--data_dir', type=str, default='/home/data', metavar='N')
    parser.add_argument('--output_dir', type=str, default='/home/models', metavar='N')
    parser.add_argument('--verbose','--v', action='store_true')

    args = parser.parse_args()

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
    args.timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    filters = args.unet_channels if args.model == 'DM_UNET' else args.nf
    substring = f'DDPM[{args.ddpm_steps}]_nf[{filters}]_{args.timestamp}'

    if args.poison_Narcissus:
        args.output_dir = os.path.join(args.output_dir,args.model,f'{args.dataset}_NS[num={args.poison_amount}_size={args.noise_sz_narcissus}_eps={args.noise_eps_narcissus}]',substring)
    else:
        args.output_dir = os.path.join(args.output_dir,args.model,args.dataset,substring)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Print args into txt file in output directory
    with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')


    # Start Training
    start_time = time.time()
    xmp.spawn(_map_train, args=(args,), nprocs=8, start_method='fork')
    print(f'Training Complete! Time taken: {time.time() - start_time} seconds')