from PureDefense import PureDefense

import os
import argparse

try: 
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except: pass

from utils.utils import *

from utils.utils_purify import get_poisons, ImageListDataset, save_poisons, process_args


### Main Function ###
def main(rank, args):

    # Set the device and seed (if not None)
    device = get_device(args.device_type)
    set_seed(args.seed, device, args.device_type)

    # Process the arguments for each rank
    args = process_args(args,rank)
    if args is None:
        xm.rendezvous('training end!')
        return
    
    # Get the data loader and number of target indices
    if args.poison_type is None:
        target_indices = 1
        purify_pbar = True
    else:
        if args.poison_type == 'Narcissus':
            target_indices = 10
        elif args.poison_type == 'Gradient_Matching':
            target_indices = 100
        elif args.poison_type == 'BullseyePolytope':
            if args.num_images_bp == 50:
                target_indices = 48 # 48 images in the dataset from original paper
            else:
                target_indices = 50
        elif args.poison_type == 'BullseyePolytope_Bench':
            target_indices = 100

        purify_pbar = False

    # Get diff and ebm model paths
    if args.ebm_model is not None: ebm_path = os.path.join(args.data_dir,'models',args.ebm_model,args.ebm_name+'.pt')
    else: ebm_path = None
    if args.diff_model is not None: diff_path = os.path.join(args.data_dir,'models',args.diff_model,args.diff_name+'.pt')
    else: diff_path = None

    # Create the PureDefense object
    PurifyClass = PureDefense(device,args.device_type,
                            ebm_type=args.ebm_model,ebm_path=ebm_path,ebm_nf=args.ebm_nf,
                            diff_type=args.diff_model,diff_path=diff_path,diff_nf=args.diff_nf,
                            diff_schedule=args.diff_schedule,
                            diff_train_steps=args.diff_train_steps,diff_output=args.diff_output,
                            img_sz=32,verbose=args.verbose)
    
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
            elif args.dataset == 'stl10_64':
                train_data = torchvision.datasets.STL10(root=args.data_dir, split='train', download=(not os.path.exists(os.path.join(args.data_dir, 'stl10_binary'))), transform=torchvision.transforms.Compose([torchvision.transforms.Resize(64),torchvision.transforms.ToTensor()]))
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4)
        else:
            poison_tuple_list, poison_indices, target_mask_label = get_poisons(args,args.target_index)
            train_loader = torch.utils.data.DataLoader(ImageListDataset(poison_tuple_list), batch_size=128, shuffle=False, num_workers=4)

        ### Purify the dataset ###
        purified_data = PurifyClass.purify(train_loader,ebm_lang_steps=args.ebm_lang_steps,ebm_lang_temp=args.ebm_lang_temp,
                        diff_steps=args.diff_purify_steps, diff_eta=args.diff_eta,
                        purify_reps=1,pbar=purify_pbar)
        

        ### Save the purified data ###
        data_key = ''
        if args.ebm_lang_steps > 0 and args.ebm_model is not None:
            data_key += f'{args.ebm_model}[{args.ebm_name}_nf{args.ebm_nf}]_{args.ebm_lang_steps}Steps_T{args.ebm_lang_temp}'
        if args.diff_purify_steps > 0 and args.diff_model is not None:
            data_key += f'_{args.diff_model}[{args.diff_name}_nf{args.diff_nf}]_beta[{args.diff_train_steps}_{args.diff_schedule}]_{args.diff_purify_steps}Steps_{args.diff_eta}eta'
        if args.ebm_lang_steps > 0 and args.diff_purify_steps > 0 and args.purify_reps > 1 and args.ebm_model is not None and args.diff_model is not None:
            data_key += f'_reps{args.purify_reps}'
        
        if data_key == '':
            data_key = 'Baseline'

        if args.poison_type is None:
            if not os.path.exists(os.path.join(args.data_dir,'PureDefense',args.dataset)):
                os.makedirs(os.path.join(args.data_dir,'PureDefense',args.dataset))
            torch.save(purified_data,os.path.join(args.data_dir,'PureDefense',args.dataset,f'{data_key}.pt'))
        else:
            save_dir = save_poisons(args,purified_data, poison_indices, target_mask_label, data_key)

        if purify_pbar is False and rank == 0:
            # Update and set description
            pbar.update(1)
            pbar.set_description(f'Poisoned Data Saved: {data_key}')

    if purify_pbar is False and rank == 0:
        pbar.close()

    # Renendezvous
    xm.rendezvous('training end!')


### Initializer ###
if __name__ == '__main__':
    
    os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'

    # ======== arg parser =================================================
    parser = argparse.ArgumentParser(description='PyTorch Poison Attack')

    ### Setup Arguments ###
    parser.add_argument('--remote_user', type=str, help='username for the remote server (TPU only, else pass in full directory args below)')
    # parser.add_argument('--num_proc', type=int, default=8, help='number of processes for TPU')
    parser.add_argument('--device_type', default='xla', type=str, choices=['xla','cuda','cpu','mps'],help='device type to use')
    parser.add_argument('--seed', default=11, type=int,help='seed for reproducibility')
    parser.add_argument('--verbose','--v', default=False, action='store_true',help='print out additional information when running')
    parser.add_argument('--data_dir', default='/home/data/', type=str, help='path to the data directory')
    parser.add_argument('--num_proc', type=int, default=1, help='number of processes for TPU')

    ### Experiment Arguments ###
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10','cinic10','stl10','stl10_64','tinyimagenet'],help='dataset to use')

    ### Purification Arguments ###

    parser.add_argument('--purify_reps', default=1, type=int, help='number of purification repetitions (when using both EBM and Diffusion)')

    # EBM Arguments 
    args_ebm = parser.add_argument_group('EBM')
    args_ebm.add_argument('--ebm_model', default='EBMSNGAN32', type=none_or_str, choices=[None,'SuperLightEBM','LightEBM','EBM','EBMSNGAN32','EBMSNGAN128','EBMSNGAN256'],help='type of EBM model to use')
    args_ebm.add_argument('--ebm_name', default='ebm_cifar10_45k', type=str_or_str_list, help='path to the EBM model including train dataset')
    args_ebm.add_argument('--ebm_nf', default=128, type=int_or_int_list,  help='number of filters for the ebm model')
    args_ebm.add_argument('--ebm_lang_steps', default=150, type=int_or_int_list, help='number of langevin steps')
    args_ebm.add_argument('--ebm_lang_temp', default=1e-4, type=float_or_float_list, help='langevin temperature')

    # Diffusion Arguments
    args_diff = parser.add_argument_group('Diffusion')
    args_diff.add_argument('--diff_model', default='DDPM_UNET_EBM', type=none_or_str, choices=[None, 'DDPM_UNET','DDPM_UNET_EBM'],help='type of diffusion model to use')
    args_diff.add_argument('--diff_name', default='mcmc_steps1600_bank_fixer_cosine', type=str_or_str_list, help='path to the diffusion model')
    args_diff.add_argument('--diff_nf', default=128, type=int_or_int_list,  help='number of filters for the unet model')
    args_diff.add_argument('--diff_train_steps', default=1000, type=int_or_int_list, help='training t-steps for diffuion model')
    args_diff.add_argument('--diff_output', default='epsilon', type=str, choices=['epsilon','start_x'],  help='diffusion model output')
    args_diff.add_argument('--diff_schedule', default='cosine', type=str, choices=['linear','cosine'], help='t schedule')
    args_diff.add_argument('--diff_purify_steps', default=10, type=int_or_int_list,  help='number of purify t-steps for the unconditional diffuion model')
    args_diff.add_argument('--diff_eta', default=0, type=int_or_int_list,  help='ddpm 1 or ddim 0 for the sampling of the 1000 tstep fixer')
        

    ### Poison Arguments ###
    parser.add_argument('--poison_type', default=None, type=str, choices=['Narcissus', 'Gradient_Matching','BullseyePolytope','BullseyePolytope_Bench'],help='type of poison to generate')
    parser.add_argument('--poison_mode', default='from_scratch', type=str, choices=['from_scratch','transfer'],help='mode of attack')
    parser.add_argument('--noise_sz_narcissus', default=32, type=int, help='size of the noise trigger for Narcissus')
    parser.add_argument('--noise_eps_narcissus', default=8, type=int, help='epsilon for the noise trigger for Narcissus')
    parser.add_argument('--num_images_narcissus', default=500, type=int, help='number of poisoned images generated')
    parser.add_argument('--random_imgs_narcissus', default=False, action='store_true', help='use random images for narcissus')
    parser.add_argument('--iters_bp', default=800, type=int,help='iterations for making poison')
    parser.add_argument('--num_images_bp', default=50, type=int,help='number of poisoned images generated')
    parser.add_argument('--net_repeat_bp', default=1, type=int, help='number of times to repeat the network for methods BP-1, BP-3, BP-5')
    parser.add_argument('--num_per_class_bp', default=50, type=int, help='num of samples per class for re-training, or the poison dataset')

    # Parse the arguments
    args = parser.parse_args()

    # Raise errors
    if args.purify_reps > 1 and (args.ebm_type is None or args.diff_type is None):
        raise ValueError('When purify_reps>1, both EBM and Diffusion models must be provided')
    if args.purify_reps > 1 and (args.ebm_lang_steps == 0 or args.diff_purify_steps == 0):
        raise ValueError('When purify_reps>1, ebm_lang_steps and diff_purify_steps must be greater than 0')
    
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
