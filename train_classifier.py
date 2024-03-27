import os
import json
import argparse
import configparser
import time
from tqdm import tqdm
import sys

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn

try: 
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except: pass

from utils.utils import *
from utils.utils_clf import *
from utils.utils_baselines import *

def main(rank, args):

    ##############################
    # Setup
    ##############################

    # Set the device and seed (if not None)
    device = get_device(args.device_type)
    set_seed(args.seed, device, args.device_type)

    # Set the poison target index (image or class label) and check if training index is out of bounds (for TPU)
    target_index = set_target_index_and_check_end(args, rank)
    if target_index is None:
        xm.rendezvous('training end!')
        return
    
    if args.verbose: print(f'Running on {xm.get_ordinal()} with rank {rank} and target index {target_index} and rand {torch.rand(1)}')

    ######################
    # Load Training Data #
    ######################

    test_trigger_loaders,poison_target_image, target_mask_label = None,None,None

    train_transforms = get_train_transforms(args)

    train_data, target_mask_label = get_base_poisoned_dataset(args,target_index,train_transforms,device)

    if 'HLB' in args.model and args.dataset in ['cifar10']:
        train_loader = train_data
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,num_workers=4)

    # Print training data details
    if args.verbose:
        p_count = sum(p.sum().item() for _, _, _, p in train_loader) 
        if 'HLB' in args.model and args.dataset in ['cifar10']: print(f'Loaded training data {len(train_loader.images)} samples, {p_count} poisoned or {p_count/len(train_loader.images):.2%} poisoned')
        else: print(f'Loaded training data {len(train_loader.dataset)} samples, {p_count} poisoned or {p_count/len(train_loader.dataset):.2%} poisoned')

    if args.baseline_defense == 'Friendly':
        
        if args.poison_mode == 'from_scratch':
            if args.dataset in ['cifar10','cinic10']:
                train_transforms_no_augs = transforms.Compose([transforms.Normalize(mean=cifar_mean_gm, std=cifar_std_gm)])
            elif args.dataset in ['stl10','stl10_64']:
                train_transforms_no_augs = transforms.Compose([transforms.Normalize(mean=stl10_mean, std=stl10_std)])
            elif args.dataset == 'tinyimagenet':
                train_transforms_no_augs = transforms.Compose([transforms.Normalize(mean=tinyimagenet_mean, std=tinyimagenet_std)])
            else:
                raise ValueError('Friendly Defense not supported for this dataset')
            train_data_no_augs, _ = get_base_poisoned_dataset(args,target_index,train_transforms_no_augs,device)
        else:
            train_transforms_no_augs = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar_mean, std=cifar_std)])
            train_data_no_augs, _ = get_base_poisoned_dataset(args,target_index,train_transforms_no_augs,device)

        train_loader_noaugs = torch.utils.data.DataLoader(train_data_no_augs, batch_size=args.batch_size, shuffle=True,num_workers=4)

    ##########################
    # Load Test/Target Data  #
    ##########################
        
    if args.poison_mode == 'from_scratch':
        if args.dataset == 'cifar10':
            test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar_mean_gm, std=cifar_std_gm)])
        elif args.dataset == 'cinic10':
            test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar_mean_gm, std=cifar_std_gm)])
        elif args.dataset == 'stl10':
            test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=stl10_mean, std=stl10_std)])
        elif args.dataset == 'stl10_64':
            test_transforms = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(), transforms.Normalize(mean=stl10_mean, std=stl10_std)])
        elif args.dataset == 'tinyimagenet':
            test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=tinyimagenet_mean, std=tinyimagenet_std)])
    else:
        test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar_mean, std=cifar_std)])

    test_data = get_test_dataset(args, test_transforms)

    if 'HLB' in args.model and args.dataset in ['cifar10']:
        test_loader = CifarLoader(test_data, train=False, batch_size=1000,dataset_name=args.dataset)
    else:
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=128,num_workers=4)

    if args.dataset == 'cinic10':
        cifar_test_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=(not os.path.exists(os.path.join(args.data_dir, 'cifar-10-batches-py'))), transform=test_transforms)
        if 'HLB' in args.model:
            cifar_test_loader = CifarLoader(cifar_test_data, train=False, batch_size=1000)
        else:
            cifar_test_loader = torch.utils.data.DataLoader(cifar_test_data, batch_size=128,num_workers=4)

    if not args.no_poison:
        if args.poison_type == 'Narcissus':
            test_trigger_loaders = get_poisons_target(args, target_index, test_transforms, target_mask = target_mask_label)
        else:
            poison_target_image, target_orig_label = get_poisons_target(args, target_index, test_transforms)

    if args.verbose: 
        if 'HLB' in args.model and args.dataset in ['cifar10']: print(f'Loaded the test data with poison type {args.poison_type}, length {len(test_loader.images)}')
        else: print(f'Loaded the test data with poison type {args.poison_type}, length {len(test_loader.dataset)}')

    ##############################
    # Load the target network, optimizer, and loss function
    ##############################

    # Load the target network
    target_net = load_target_network(args,device)

    if args.verbose: print(f'Loaded target network {args.model} with num params: {sum(p.numel() for p in target_net.parameters())}')

    # Optimizer
    optimizer = get_optimizer(args,target_net)

    # Scheduler
    if 'HLB' in args.model:
        if args.dataset == 'cifar10':
            total_train_steps = np.ceil(len(train_data) * args.epochs)
        else:
            total_train_steps = np.ceil(len(train_data) / args.batch_size * args.epochs) + args.epochs
        lr_schedule = np.interp(np.arange(1+total_train_steps),
                            [0, int(0.2 * total_train_steps), total_train_steps],
                            [0.2, 1, 0]) # triangular learning rate schedule
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
        if args.verbose: print(f'Loaded the HLB scheduler with {total_train_steps} steps')

    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay, gamma=0.1)

    # Loss function (only used for non-HLB models)
    if 'HLB' in args.model:
        criterion = nn.CrossEntropyLoss(reduction='none',label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    ##############################
    # Run the attack (Training Loop)
    ##############################

    # Initialize tqdm on master device only
    if (args.device_type == 'xla' and xm.is_master_ordinal()) or args.device_type != 'xla':
        pbar = tqdm(total=args.epochs)

    # Training Logs
    logs = {'train_loss': [], 'test_acc': []}

    if not args.no_poison: 
        logs['p_acc'] = []
        if args.poison_type == 'Narcissus': 
            logs['t_acc'] = []

    if args.dataset == 'cinic10': logs['cifar_acc'] = []

    if args.baseline_defense == 'Epic':
        times_selected = torch.zeros(len(train_data), dtype=torch.int32)
        base_loader = train_loader
        logs['subset_size'] = {}

    train_start_time = time.time()

    target_net.train()

    for epoch in range(args.epochs):

        # _______________________________________________________________________________________
        # Craft Friendly Noise if Friendly Defense
        if args.baseline_defense == 'Friendly' and 'friendly' in args.friendly_noise_type and epoch == args.friendly_begin_epoch:
            friendly_noise = generate_friendly_noise(target_net,train_loader_noaugs,device,args.device_type,friendly_epochs=args.friendly_epochs,mu=args.friendly_mu,
                                                        friendly_lr=args.friendly_lr, clamp_min=-args.friendly_clamp / 255, clamp_max=args.friendly_clamp / 255,model_train=True,img_dim = dataset_dict[args.dataset]['img_dim']
                                                    )
            target_net.zero_grad()
            if args.device_type == 'xla': xm.mark_step()
            if args.verbose: print(f"Friendly noise stats:  Max: {torch.max(friendly_noise)}  Min: {torch.min(friendly_noise)}  Mean (abs): {torch.mean(torch.abs(friendly_noise))}  Mean: {torch.mean(friendly_noise)}")
            train_data.set_perturbations(friendly_noise)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,num_workers=4)
        # _______________________________________________________________________________________
        # _______________________________________________________________________________________
        # Epic Defense
        if ((args.baseline_defense == 'Epic') and (args.epic_subset_size < 1) and (epoch % args.epic_subset_freq == 0) and (epoch >= args.epic_drop_after) and (epoch <= args.epic_stop_after)):

            train_loader = run_epic(args, target_net, base_loader, epoch, device, times_selected)
            
            if args.verbose: print(f'New training set size: {len(train_loader.dataset)}')

            logs['subset_size'][epoch] = len(train_loader.dataset)
        # _______________________________________________________________________________________
        
        # Train the model
        logs['train_loss'].append(0)

        target_net.train()

        for input, target, index, p in train_loader:
            
            input, target = input.to(device), target.to(device)
            output = target_net(input)

            # Backward pass
            optimizer.zero_grad()
            loss = criterion(output, target)
            if 'HLB' in args.model:
                loss.sum().backward()
                optimizer.step()
                scheduler.step()
                logs['train_loss'][-1] += loss.mean().item()
            else:
                loss.backward()
                optimizer.step()
                logs['train_loss'][-1] += loss.item()

            if args.device_type == 'xla': xm.mark_step()

            if args.baseline_defense == 'Epic':
                times_selected[index] += 1

        # Update logs
        logs['train_loss'][-1] /= len(train_loader)
        
        # Decay the learning rate
        if 'HLB' not in args.model: scheduler.step()
        if args.device_type == 'xla': xm.mark_step()

        # Test the model for from_scratch attacks
        logs = eval_epoch(args,target_net, logs, test_loader, device,
                          test_trigger_loaders = test_trigger_loaders,
                          poison_target_image = poison_target_image,
                            target_mask_label = target_mask_label,
                            target_index = target_index,
                            )
        
        # Update progress bar
        if (args.device_type == 'xla' and xm.is_master_ordinal()) or args.device_type != 'xla':
            update_progress_bar(args, pbar, epoch, logs)

    training_time = time.time() - train_start_time

    # Get the final test accuracy and poison success
    end_acc = get_test_acc(target_net, test_loader, device)

    if args.dataset == 'cinic10':
        cifar_end_acc = get_test_acc(target_net, cifar_test_loader, device)

    if not args.no_poison:

        if args.poison_type != 'Narcissus':
            img_dim = dataset_dict[args.dataset]['img_dim']
            target_pred = target_net(poison_target_image.to(device).view(1,3,img_dim,img_dim))
            pred = torch.argmax(target_pred).item()
            success = bool(pred == target_mask_label)
            correct_class = bool(pred == target_orig_label)
                       
        else:
            p_accs = {}
            t_accs = {}
            for i in range(1,4):
                _, p_acc, t_acc = run_test_epoch_narcissus(test_trigger_loaders[i], target_net, nn.CrossEntropyLoss(reduction='none'),target_index, device)
                p_accs[i] = p_acc
                t_accs[i] = t_acc

    ##############################
    # Save the results and models
    ##############################
                
    if args.no_poison:
        get_accs_save_results_clean(args, rank, target_index, end_acc, training_time, logs)
    else:
        if args.poison_type != 'Narcissus':
            get_accs_save_results(args, rank, target_index, end_acc, success, correct_class, training_time, logs)
        else:
            get_accs_save_results_Narcissus(args, rank, target_index, end_acc, training_time, logs, p_accs, t_accs)

    # Save the model
    if args.save_models:
        model_save_dir = os.path.join(args.output_dir,'Models',f'{args.model}',f'{args.data_key}',f'{args.experiment_timestamp}_Index_{target_index}')
        if os.path.exists(model_save_dir) == False: os.makedirs(model_save_dir)
        torch.save(target_net.to('cpu').state_dict(), os.path.join(model_save_dir,'model.pt'))

    # Print the results
    if args.no_poison:
        print(f'Index {target_index} | Training Time {training_time} | End Acc {end_acc:.2%}')
    else:
        if args.device_type == 'xla' and xm.is_master_ordinal():
            print(f'Data Key {args.data_key}') 
        if args.poison_type != 'Narcissus':
            print(f'Index {target_index} | Poison Type {args.poison_type} | End Acc {end_acc:.2%} | Poison Success {success} | Correct Pred {correct_class} | Training Time {training_time}')
        elif args.poison_type == 'Narcissus':
            if args.dataset == 'cinic10': # TODO Fix cifar_acc
                print(f'Index {target_index} | Poison Type {args.poison_type} | P1 Acc {p_accs[1]:.2%} | T1 Acc {t_accs[1]:.2%} | End Acc {end_acc:.2%} | CIFAR Acc {cifar_end_acc:.2%} | Training Time {training_time:.1f} ')
            else: 
                print(f'Index {target_index} | Poison Type {args.poison_type} | P1 Acc {p_accs[1]:.2%} | T1 Acc {t_accs[1]:.2%} | End Acc {end_acc:.2%} | Training Time {training_time:.1f}')
        
    # Rendezvous 
    if args.device_type == 'xla': xm.rendezvous('training end!')
    
    # Concat the dataframes
    if args.device_type == 'xla' and xm.is_master_ordinal():
        concat_result_dataframes_xla(args)


if __name__ == '__main__':
    
    os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'

    # ======== arg parser =================================================
    parser = argparse.ArgumentParser(description='PyTorch Poison Attack')

    ### Setup Arguments ###
    parser.add_argument('--remote_user', type=str, help='username for the remote server (TPU only, else pass in full directory args below)')
    parser.add_argument('--num_proc', type=int, default=8, help='number of processes for TPU')
    parser.add_argument('--config_file', default='./Configs/config.ini', type=str, help='path to the config file')
    parser.add_argument('--config_override', default=None, type=str, help='use this to override the specific config settings with a ini section')
    parser.add_argument('--seed', default=11, type=int,help='seed for reproducibility')
    parser.add_argument('--save_models', default=False, action='store_true',help="Whether to save the models")
    parser.add_argument('--exp_name', default=None, type=str,help='name of the experiment to append to the output dataframe')
    parser.add_argument('--no_poison', default=False, action='store_true',help='whether to run the attack or not')
    parser.add_argument('--start_target_index', default=0, type=int,help='start label for the attack (only used for from_scratch attacks)')
    parser.add_argument('--data_dir', default='/home/data/', type=str, help='path to the data directory')
    parser.add_argument('--output_dir', default='/home/results_PureGen_PoisonDefense/', type=str, help='path to the output directory')
    parser.add_argument('--verbose','--v', default=False, action='store_true',help='print out additional information when running')

    ### Experiment Arguments ###
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10','cinic10','stl10','stl10_64','tinyimagenet'],help='dataset to use')
    parser.add_argument('--data_key', default='Baseline', type=str, help='key for the purified or baseline data')
    parser.add_argument('--model', default='HLB', type=str, choices=['HLB','ResNet18_HLB','ResNet18','ResNet34','MobileNetV2','DenseNet121'],help='type of model to use')
    parser.add_argument('--poison_mode', default='from_scratch', type=str, choices=['from_scratch','transfer'],help='mode of attack')
    parser.add_argument('--poison_type', default='Narcissus', type=str, choices=['Narcissus', 'GradientMatching','BullseyePolytope','BullseyePolytope_Bench'],help='type of poison to generate')
    parser.add_argument('--fine_tune', default=False, action='store_true',help="Whether retrain the full model (fine-tuning) or just the linear layer (default: False)")
    parser.add_argument('--baseline_defense', default='None', type=str, choices=['None','Epic','Friendly'],help='type of defense to use')
    parser.add_argument('--selected_indices', default=None, nargs='+', type=int, help='Specific indices to run the attack on each TPU core (default: None, TPU only!!!)')
    
    ### Poison Arguments ###
    parser.add_argument('--noise_sz_narcissus', default=32, type=int, help='size of the noise trigger for Narcissus')
    parser.add_argument('--noise_eps_narcissus', default=8, type=int, help='epsilon for the noise trigger for Narcissus')
    parser.add_argument('--num_images_narcissus', default=500, type=int_or_int_list, help='number of poisoned images generated')
    parser.add_argument('--iters_bp', default=800, type=int,help='iterations for making poison')
    parser.add_argument('--num_images_bp', default=50, type=int,help='number of poisoned images generated')
    parser.add_argument('--net_repeat_bp', default=1, type=int, help='number of times to repeat the network for methods BP-1, BP-3, BP-5')
    parser.add_argument('--num_per_class_bp', default=50, type=int, help='num of samples per class for re-training, or the poison dataset')

    ### Baseline Defense Arguments ###
    parser.add_argument('--friendly_noise_type', default=['friendly','bernoulli'], type=str, nargs='*', help='type of noise to apply', choices=["uniform", "gaussian", "bernoulli", "gaussian_blur", "friendly"])
    parser.add_argument('--epic_subset_size', type=float, help='size of the subset', default=0.1)
    parser.add_argument('--epic_drop_after', type=int, help='epoch to start dropping', default=10)


    # Parse the arguments
    args = parser.parse_args()

    ##############
    # Config File (and override) Arguments
    ##############

    config = configparser.ConfigParser()
    config.read(args.config_file)

    set_args_from_config(args, config, 'DEFAULTS')

    if args.poison_mode == 'transfer' and not args.fine_tune:
        set_args_from_config(args, config, 'TRANSFER')
    elif args.poison_mode == 'transfer' and args.fine_tune:
        set_args_from_config(args, config, 'FINE_TUNE')

    if args.config_override is not None:
        set_args_from_config(args, config, args.config_override)

    ##############
    # Error Checking
    ##############

    if args.poison_type == 'GradientMatching' and args.poison_mode == 'transfer':
        raise ValueError('Gradient Matching does not support transfer attacks')
    if args.poison_type == 'BullseyePolytope_Bench' and args.poison_mode == 'from_scratch':
        raise ValueError('BullseyePolytope_Bench does not support from_scratch attacks')
    if args.poison_type == 'BullseyePolytope' and args.poison_mode == 'from_scratch':
        raise ValueError('BullseyePolytope does not support from_scratch attacks')
    if args.selected_indices is not None and args.device_type != 'xla':
        raise ValueError('selected_indices only supported for TPU')
    if args.selected_indices is not None and args.poison_mode != 'from_scratch':
        raise ValueError('selected_indices only supported for from_scratch attacks')
    if args.baseline_defense != 'None' and args.data_key != 'Baseline':
        raise ValueError('Baseline defenses only supported for baseline data')

    ##############
    # Directories
    ##############

    # Setup directories for remote server
    if args.remote_user is not None:
        args.data_dir = args.data_dir.replace('/home',f'/home/{args.remote_user}')
        args.output_dir = args.output_dir.replace('/home',f'/home/{args.remote_user}')

    # Create the output directory and get the timestamp
    args.experiment_timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    if args.no_poison:
        args.output_dir = os.path.join(args.output_dir, 'Clean')
    else:
        if args.poison_mode == 'from_scratch':
            sub_folder = args.poison_mode.title()
        else:
            if args.fine_tune:
                sub_folder = f'{args.poison_mode.title()}_FineTune'
            else:
                sub_folder = f'{args.poison_mode.title()}_Linear'
        args.output_dir = os.path.join(args.output_dir, sub_folder, args.poison_type)
    if os.path.exists(args.output_dir) == False: os.makedirs(args.output_dir)

    # Print the arguments
    if args.verbose:
        print('\n_____Arguments_____\n')
        for arg in vars(args):
            print(f'{arg}: {getattr(args, arg)}')
        print('___________________\n')

    ##############
    # Run the attack
    ##############
        
    start_time = time.time()
    
    # If not poison from scratch, run the attack on all the targets, else run on the specified targets
    if args.poison_mode == 'from_scratch':
    
        if args.device_type == 'xla' and args.num_proc > 1:
            if args.poison_type == 'Narcissus' and args.selected_indices is None:
                for args.start_target_index in [0,8]:
                    xmp.spawn(main, args=(args,), nprocs=args.num_proc, join=True, start_method='fork')
            elif args.poison_type == 'GradientMatching' and args.selected_indices is None:
                for args.start_target_index in range(0,100,8):
                    xmp.spawn(main, args=(args,), nprocs=args.num_proc, join=True, start_method='fork')
            else:
                xmp.spawn(main, args=(args,), nprocs=args.num_proc, join=True, start_method='fork')
        else:
            main(0, args)
    else:
        if args.poison_type == 'BullseyePolytope':
            if args.num_images_bp == 50:
                poison_targets = 48 # 48 images in the dataset from original paper
            else:
                poison_targets = 50
        elif args.poison_type == 'BullseyePolytope_Bench': poison_targets = 100
        elif args.poison_type == 'Narcissus': poison_targets = 10

        if args.device_type == 'xla' and args.num_proc > 1:
            for i in range(0,poison_targets,args.num_proc):
                args.start_target_index = i
                xmp.spawn(main, args=(args,), nprocs=args.num_proc, join=True, start_method='fork')
        else:
            for i in range(0,poison_targets):
                args.start_target_index = i
                main(0, args)

    print(f"Total time taken: {time.time() - start_time}")
