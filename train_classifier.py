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
from utils.EBM_models import create_ebm


def main(rank, args):

    ##############################
    # Setup
    ##############################

    # Set the device and seed (if not None)
    device = get_device(args.device_type)
    if args.poison_type == 'NeuralTangent': set_seed(args.seed+rank, device, args.device_type)
    else: set_seed(args.seed, device, args.device_type)

    # Set the poison target index (image or class label) and check if training index is out of bounds (for TPU)
    target_index = set_target_index_and_check_end(args, rank)
    if target_index is None:
        xm.rendezvous('training end!')
        return
    
    if args.verbose: print(f'Running on {xm.get_ordinal()} with rank {rank} and target index {target_index} and rand {torch.rand(1)}')

    ######################
    # Load Training Data #
    ######################

    if args.ebm_filter is not None:
        ebm_model = create_ebm('EBMSNGAN32',128)
        ebm_model.load_state_dict(torch.load(os.path.join(args.data_dir,args.ebm_path)))
        ebm_model.to(device)
    else:
        ebm_model = None

    if args.baseline_defense == 'Friendly':
        train_data, train_loader, train_loader_noaugs, p_count, test_trigger_loaders,poison_target_image, target_mask_label = get_train_data(args, target_index, device)
    else:
        train_data, train_loader, p_count, test_trigger_loaders,poison_target_image, target_mask_label = get_train_data(args, target_index, device,ebm_model=ebm_model)

    if args.verbose:
        if 'HLB' in args.model and args.dataset in ['cifar10'] and args.baseline_defense == 'None': print(f'Loaded training data {len(train_loader.images)} samples, {p_count} poisoned or {p_count/len(train_loader.images):.2%} poisoned')
        else: print(f'Loaded training data {len(train_loader.dataset)} samples, {p_count} poisoned or {p_count/len(train_loader.dataset):.2%} poisoned')

    ##########################
    # Load Test/Target Data  #
    ##########################

    test_loader, test_transforms = get_test_dataset(args)

    if args.poison_type == 'NeuralTangent':
        pass
    elif args.poison_type == 'Narcissus':
        test_trigger_loaders = get_poisons_target(args, target_index, test_transforms, target_mask = target_mask_label)
    else:
        poison_target_image, target_orig_label = get_poisons_target(args, target_index, test_transforms)

    if args.verbose: 
        if 'HLB' in args.model and args.dataset in ['cifar10'] and args.baseline_defense == 'None': print(f'Loaded the test data with poison type {args.poison_type}, length {len(test_loader.images)}')
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
    scheduler = get_scheduler(args,optimizer,len(train_data))

    # Loss function
    if 'HLB' in args.model: criterion = nn.CrossEntropyLoss(reduction='none',label_smoothing=args.label_smoothing)
    else: criterion = nn.CrossEntropyLoss()

    ##############################
    # Run the attack (Training Loop)
    ##############################

    # Initialize tqdm on master device only
    if (args.device_type == 'xla' and xm.is_master_ordinal()) or args.device_type != 'xla':
        pbar = tqdm(total=args.epochs)

    # Training Logs
    logs = {'train_loss': [], 'test_acc': []}

    if args.poison_type != 'NeuralTangent':
        logs['p_acc'] = []
    if args.poison_type == 'Narcissus': 
        logs['t_acc'] = []

    if args.baseline_defense == 'Epic':
        if 'HLB' in args.model and args.dataset in ['cifar10']:
            times_selected = torch.zeros(len(train_loader.images), dtype=torch.int32)
        else:
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

    if args.poison_type != 'Narcissus' and args.poison_type != 'NeuralTangent':
        img_dim = dataset_dict[args.dataset]['img_dim']
        target_pred = target_net(poison_target_image.to(device).view(1,3,img_dim,img_dim))
        pred = torch.argmax(target_pred).item()
        success = bool(pred == target_mask_label)
        correct_class = bool(pred == target_orig_label)
                    
    elif args.poison_type == 'Narcissus':
        p_accs = {}
        t_accs = {}
        for i in range(1,3):
            _, p_acc, t_acc = run_test_epoch_narcissus(test_trigger_loaders[i], target_net, nn.CrossEntropyLoss(reduction='none'),target_index, device)
            p_accs[i] = p_acc
            t_accs[i] = t_acc

    ##############################
    # Save the results and models
    ##############################
                
    if args.poison_type == 'NeuralTangent':
        get_accs_save_results(args, rank, target_index, end_acc, training_time, logs)
    else:
        if args.poison_type != 'Narcissus':
            get_accs_save_results_untriggered(args, rank, target_index, end_acc, success, correct_class, training_time, logs)
        else:
            get_accs_save_results_triggered(args, rank, target_index, end_acc, training_time, logs, p_accs, t_accs)

    # Save the model
    if args.save_models:
        model_save_dir = os.path.join(args.output_dir,'Models',f'{args.model}',f'{args.data_key}',f'{args.experiment_timestamp}_Index_{target_index}')
        if os.path.exists(model_save_dir) == False: os.makedirs(model_save_dir)
        torch.save(target_net.to('cpu').state_dict(), os.path.join(model_save_dir,'model.pt'))

    # Print the results
    if args.poison_type == 'NeuralTangent':
        print(f'Index {target_index} | Training Time {training_time} | End Acc {end_acc:.2%}')
    else:
        if args.device_type == 'xla' and xm.is_master_ordinal():
            print(f'Data Key {args.data_key}') 
        if args.poison_type != 'Narcissus':
            print(f'Index {target_index} | Poison Type {args.poison_type} | End Acc {end_acc:.2%} | Poison Success {success} | Correct Pred {correct_class} | Training Time {training_time}')
        elif args.poison_type == 'Narcissus':
            print(f'Index {target_index} | Poison Type {args.poison_type} | P1 Acc {p_accs[1]:.2%} | T1 Acc {t_accs[1]:.2%} | End Acc {end_acc:.2%} | Training Time {training_time:.1f}')
        
    # Rendezvous pho
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
    parser.add_argument('--config_overrides', default=None, nargs='+', type=str, help='Config Overrides (will execute in order)')
    parser.add_argument('--exp_name', default=None, type=str,help='name of the experiment to append to the output dataframe')
    parser.add_argument('--start_target_index', default=0, type=int,help='start label for the attack (only used for from_scratch attacks)')
    parser.add_argument('--selected_indices', default=None, nargs='+', type=int, help='Specific indices to run the attack on each TPU core (default: None, TPU only!!!)')
    parser.add_argument('--verbose','--v', default=False, action='store_true',help='print out additional information when running')
    parser.add_argument('--data_dir', default='/home/data/', type=str, help='path to the data directory')
    parser.add_argument('--output_dir', default='/home/results_PureGen_PoisonDefense/', type=str, help='path to the output directory')

    ### Experiment Arguments ###
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10','cinic10','stl10','tinyimagenet'],help='dataset to use')
    parser.add_argument('--data_key', default='Baseline', type=str, help='key for the purified or baseline data')
    parser.add_argument('--model', default='HLB_S', type=str, choices=['HLB_S','HLB_M','HLB_L','ResNet18_HLB','ResNet18','ResNet34','MobileNetV2','DenseNet121'],help='type of model to use')
    parser.add_argument('--poison_mode', default='from_scratch', type=str, choices=['from_scratch','clean','linear_transfer','fine_tune_transfer'],help='mode of attack')
    parser.add_argument('--poison_type', default='Narcissus', type=str, choices=['Narcissus','NeuralTangent','GradientMatching','BullseyePolytope','BullseyePolytope_Bench'],help='type of poison to generate')
    parser.add_argument('--baseline_defense', default='None', type=str, choices=['None','JPEG','Epic','Friendly'],help='type of defense to use')

    ### EBM Filter Arguments ###
    parser.add_argument('--ebm_filter', default=None, type=float, help='EBM highest energy % to purify')
    parser.add_argument('--ebm_path', default='PureGen_Models/EBMSNGAN32/cinic10_imagenet_nf[128].pt', type=str, help='path to the EBM model')
    
    ### Poison Arguments ###
    parser.add_argument('--noise_sz_narcissus', default=32, type=int, help='size of the noise trigger for Narcissus')
    parser.add_argument('--noise_eps_narcissus', default=8, type=int, help='epsilon for the noise trigger for Narcissus')

    ### Baseline Defense Arguments ###
    parser.add_argument('--friendly_noise_type', default=['friendly','bernoulli'], type=str, nargs='*', help='type of noise to apply', choices=["uniform", "gaussian", "bernoulli", "gaussian_blur", "friendly"])
    parser.add_argument('--epic_subset_size', type=float, help='size of the subset', default=0.1)
    parser.add_argument('--epic_drop_after', type=int, help='epoch to start dropping', default=10)

    #### Parse the arguments
    args = parser.parse_args()


    ### Read the config file
    config = configparser.ConfigParser()
    config.read(args.config_file)

    # Set the arguments from the config file
    set_args_from_config(args, config, 'DEFAULTS')
    if args.poison_mode == 'linear' and not args.fine_tune: set_args_from_config(args, config, 'LINEAR_TRANSFER')
    elif args.poison_mode == 'fine_tune' and args.fine_tune: set_args_from_config(args, config, 'FINE_TUNE')
    if args.config_overrides is not None:
        for config_override in args.config_overrides:
            set_args_from_config(args, config, config_override)

    ### Error Checking
    check_arg_errors(args)

    # Get experiment timestamp
    args.experiment_timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

    # Setup the directories
    setup_directories(args)

    # Print the arguments if verbose
    if args.verbose:
        print('\n_____Arguments_____\n')
        for arg in vars(args):
            print(f'{arg}: {getattr(args, arg)}')
        print('___________________\n')

    ##################
    # Run the attack #
    ##################
        
    start_time = time.time()

    num_classifiers = poison_num_targets[args.poison_mode][args.poison_type]
    if args.device_type == 'xla' and args.num_proc > 1:
        for args.start_target_index in range(0,num_classifiers,8):
            xmp.spawn(main, args=(args,), nprocs=args.num_proc, join=True, start_method='fork')
    else:
        for args.start_target_index in range(0,num_classifiers):
            main(0, args)

    print(f"Total time taken: {time.time() - start_time}")
