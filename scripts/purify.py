from PureGen import PureGen

import os
import argparse
import time
from tqdm import tqdm
import torch
import torchvision

try: 
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except: pass


# Add parent directory to sys path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import *
from utils.purification import get_poisons, ImageListDataset, save_poisons, process_args, get_ntg, download_poisons


### Main Function ###
def main(rank, args):

    # Set the device and seed (if not None)
    device = get_device(args.device_type)
    set_seed(args.seed, device, args.device_type)

    # Process the arguments for each rank
    args = process_args(args,rank)
    if args is None: 
        xm.rendezvous('purification end!')
        return
    
    # Raise errors
    if args.purify_reps > 1 and (args.ebm_model is None or args.diff_model is None):
        raise ValueError('When purify_reps > 1, both EBM and Diffusion models must be provided')
    if args.purify_reps > 1 and (args.ebm_lang_steps == 0 or args.diff_T == 0):
        raise ValueError('When purify_reps > 1, ebm_lang_steps and diff_T must be greater than 0')

    # Get the data loader and number of target indices
    if args.poison_type in [None,'NeuralTangent','TransferBase']:
        target_indices = 1
        purify_pbar = True
    else:
        if args.poison_type == 'Narcissus':
            target_indices = 10
        elif args.poison_type == 'GradientMatching':
            target_indices = 100
        elif args.poison_type == 'BullseyePolytope':
            if args.num_images_bp == 50:
                target_indices = 48 # 48 images in the dataset from original paper
            else:
                target_indices = 50
        elif args.poison_type == 'BullseyePolytope_Bench':
            target_indices = 100

        purify_pbar = False

    # Download poisons if they don't exist
    if args.poison_type is not None:
        download_poisons(args.data_dir)

    # Create the PureGen object
    PurifyClass = PureGen(device, args.device_type,
                            ebm_path=args.ebm_path, ebm_nf=args.ebm_nf,                                              # EBM Model
                            diff_path=args.diff_path, diff_unet_channels=args.unet_channels,        # Diffusion Model
                            jpeg_compression=args.jpeg_compression,                                             # JPEG Compression
                            verbose=args.verbose)
    
    if purify_pbar is False and rank == 0:
        pbar = tqdm(total=target_indices, desc='Purifying Poisoned Data')
    
    for i,args.target_index in enumerate(range(target_indices)):

        ### Get Data to Purify ###
        if args.poison_type is None:
            if args.dataset == 'cifar10':
                train_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=(not os.path.exists(os.path.join(args.data_dir, 'cifar-10-batches-py'))), transform=torchvision.transforms.ToTensor())
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4)
            elif args.dataset == 'stl10':
                train_data = torchvision.datasets.STL10(root=args.data_dir, split='train', download=(not os.path.exists(os.path.join(args.data_dir, 'stl10_binary'))), transform=torchvision.transforms.ToTensor())
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4)
            elif args.dataset == 'cinic10':
                train_data = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'CINIC-10/train'), transform=torchvision.transforms.ToTensor())
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4)
            elif args.dataset == 'stl10_64':
                train_data = torchvision.datasets.STL10(root=args.data_dir, split='train', download=(not os.path.exists(os.path.join(args.data_dir, 'stl10_binary'))), transform=torchvision.transforms.Compose([torchvision.transforms.Resize(64),torchvision.transforms.ToTensor()]))
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4)
            elif args.dataset == 'tinyimagenet':
                train_data = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'tiny-imagenet-200/train'), transform=torchvision.transforms.ToTensor())
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4)
        elif args.poison_type == 'NeuralTangent':
            train_data = get_ntg(os.path.join(args.data_dir,'PureGen_Poisons/NTG'),True, transform=torchvision.transforms.ToTensor())
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4)
        elif args.poison_type == 'TransferBase':
            train_data = ImageListDataset(torch.load(os.path.join(args.data_dir,'CIFAR10_TRAIN_Split.pth'))['others'])
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4)
        else:
            poison_tuple_list, poison_indices, target_mask_label = get_poisons(args,args.target_index)
            train_loader = torch.utils.data.DataLoader(ImageListDataset(poison_tuple_list), batch_size=128, shuffle=False, num_workers=4)

        if args.verbose:
            print(f'Purifying Data: {args.dataset} - {args.poison_type} - {args.target_index} of size {len(train_loader.dataset)}')

        ### Purify the dataset ###
        start_purify = time.time()
        purified_data = PurifyClass.purify(train_loader,ebm_lang_steps=args.ebm_lang_steps,
                                            diff_steps=args.diff_T,
                                            purify_reps=args.purify_reps,pbar=purify_pbar)

        purify_time = time.time() - start_purify
        
        ### Create Data Key ###

        # Primary Defenses
        data_key = ''
        if args.ebm_model:
            data_key += f'EBM[Steps[{args.ebm_lang_steps}]]'
        if args.diff_model:
            data_key += f'_DDPM[T[{args.diff_T}]]'
        if args.jpeg_compression is not None:
            data_key += f'_JPEG[{args.jpeg_compression}]'
        if args.purify_reps > 1: 
            data_key += f'_reps{args.purify_reps}'

        # Baseline or strip Leading underscore
        if data_key == '':
            data_key = 'Baseline'
        if data_key[0] == '_':
            data_key = data_key[1:]

        # Save the purified data
        if args.poison_type is None:
            if not os.path.exists(os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset)):
                os.makedirs(os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset))
            torch.save(purified_data,os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset,f'{data_key}.pt'))
        elif args.poison_type == 'NeuralTangent':
            if not os.path.exists(os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset,'NTG')):
                os.makedirs(os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset,'NTG'))
            torch.save(purified_data,os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset,'NTG',f'{data_key}.pt'))
        elif args.poison_type == 'TransferBase':
            if not os.path.exists(os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset,'TransferBase')):
                os.makedirs(os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset,'TransferBase'))
            torch.save(purified_data,os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset,'TransferBase',f'{data_key}.pt'))
        else:
            save_dir = save_poisons(args,purified_data, poison_indices, target_mask_label, data_key)

        # Save the purify time
        try: save_purify_time(data_key,purify_time,args,os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset))
        except: pass

        if purify_pbar is False and rank == 0:
            # Update and set description
            pbar.update(1)
            pbar.set_description(f'Poisoned Data Saved: {data_key}')

        if args.device_type == 'xla': xm.mark_step()

    if purify_pbar is False and rank == 0:
        pbar.close()

    # Renendezvous
    xm.rendezvous('purification end!')


### Initializer ###
if __name__ == '__main__':
    
    os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'

    # ======== arg parser =================================================
    parser = argparse.ArgumentParser(description='PyTorch Poison Attack')

    ### Setup Arguments ###
    parser.add_argument('--remote_user', type=str, help='username for the remote server (TPU only, else pass in full directory args below)')
    parser.add_argument('--num_proc', type=int, default=1, help='number of processes for TPU')
    parser.add_argument('--device_type', default='xla', type=str, choices=['xla','cuda','cpu','mps'],help='device type to use')
    parser.add_argument('--seed', default=11, type=int,help='seed for reproducibility')
    parser.add_argument('--verbose','--v', default=False, action='store_true',help='print out additional information when running')
    parser.add_argument('--data_dir', default='/home/data/', type=str, help='path to the data directory')
    
    ### Experiment Arguments ###
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10','cinic10','stl10','tinyimagenet'],help='dataset to use')
    parser.add_argument('--jpeg_compression', default=None, type=int_or_int_list, help='jpeg compression quality (No compression if None)')

    ### Purification Arguments ###

    parser.add_argument('--purify_reps', default=1, type=int_or_int_list, help='number of purification repetitions (when using both EBM and Diffusion)')

    # EBM Arguments 
    args_ebm = parser.add_argument_group('EBM')
    args_ebm.add_argument('--ebm_model', action='store_true', help='use EBM model')
    args_ebm.add_argument('--ebm_path', default='SunayBhat1/puregen-ebm-cinic10-imagenet', type=str_or_str_list, help='path to the EBM model including train dataset')
    args_ebm.add_argument('--HF_Pretrained_EBM', default=True, action='store_true', help='use HF pretrained EBM')
    args_ebm.add_argument('--ebm_nf', default=128, type=int_or_int_list,  help='number of filters for the ebm model')
    args_ebm.add_argument('--ebm_lang_steps', default=150, type=int_or_int_list, help='number of langevin steps')

    # Diffusion Arguments
    args_diff = parser.add_argument_group('Diffusion')
    args_diff.add_argument('--diff_model', action='store_true', help='use diffusion model')
    args_diff.add_argument('--diff_path', default='SunayBhat1/puregen-ddpm-cinic10-imagenet', type=str_or_str_list, help='path to the diffusion model')
    args_diff.add_argument('--HF_Pretrained_Diff', default=True, action='store_true', help='use HF pretrained Diffusion')
    args_diff.add_argument('--unet_channels', default=[128, 128, 256, 256, 512, 512], type=int_or_int_list, help='number of channels for the unet model')
    args_diff.add_argument('--diff_T', default=75, type=int_or_int_list,  help='number of purify t-steps for the unconditional diffuion model')

    ### Poison Arguments ###
    parser.add_argument('--poison_type', default=None, type=str, choices=['Narcissus','GradientMatching','TransferBase','BullseyePolytope','BullseyePolytope_Bench','NeuralTangent'],help='type of poison to generate')
    parser.add_argument('--poison_mode', default='from_scratch', type=str, choices=['from_scratch','linear_transfer','fine_tune_transfer'],help='mode of attack')
    parser.add_argument('--noise_sz_narcissus', default=32, type=int, help='size of the noise trigger for Narcissus')
    parser.add_argument('--noise_eps_narcissus', default=8, type=int, help='epsilon for the noise trigger for Narcissus')
    parser.add_argument('--num_images_narcissus', default=500, type=int_or_int_list, help='number of poisoned images generated')
    parser.add_argument('--random_imgs_narcissus', default=False, action='store_true', help='use random images for narcissus')
    parser.add_argument('--iters_bp', default=800, type=int,help='iterations for making poison')
    parser.add_argument('--num_images_bp', default=50, type=int,help='number of poisoned images generated')
    parser.add_argument('--net_repeat_bp', default=1, type=int, help='number of times to repeat the network for methods BP-1, BP-3, BP-5')

    # Parse the arguments
    args = parser.parse_args()
    
    # Print the arguments
    if args.verbose:
        print('\n_____Arguments_____\n')
        for arg in vars(args):
            print(f'{arg}: {getattr(args, arg)}')
        print('___________________\n')
        
    start_time = time.time()

    # Setup directories for remote server
    if args.remote_user is not None:
        args.data_dir = args.data_dir.replace('/home',f'/home/{args.remote_user}')
    
    xmp.spawn(main, args=(args,), nprocs=args.num_proc, join=True, start_method='fork')

    print(f"Total time taken: {time.time() - start_time}")
