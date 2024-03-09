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
from torch.autograd import Variable

try: 
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except: pass

from diffusers import UNet2DModel
from diffusers import DDPMScheduler

from models import load_model
from utils.utils import *
from utils.utils_data import *
from utils.utils_ebm import *
from utils.utils_baselines import *
from utils.utils_optim import *

def main(rank, args):

    # Set the device and seed (if not None)
    device = get_device(args.device_type)
    set_seed(args.seed, device, args.device_type)

    # Set the poison target index (image or class label) and check if training index is out of bounds (for TPU)
    target_index = set_target_index_and_check_end(args, rank)
    if target_index is None:
        xm.rendezvous('training end!')
        return
    
    if args.verbose: print(f'Running on {xm.get_ordinal()} with rank {rank} and target index {target_index} and rand {torch.rand(1)}')

    ##############################
    # Setup EBM
    ##############################

    if args.defense in ['EBM','EBM_Diff']:
        ebm_model = get_ebm(args, device)
    else:
        ebm_model = None

    ##############################
    # Setup Diffusion Model
    ##############################
        
    if args.defense in ['Diff','EBM_Diff']:

        diff_model_id = "google/ddpm-cifar10-32"
        diff_model = UNet2DModel.from_pretrained(diff_model_id).to(device)

        diff_scheduler = DDPMScheduler.from_pretrained(diff_model_id)
        diff_scheduler.config['num_train_timesteps'] = args.diff_steps  # Reduced number of diffusion steps
        diff_scheduler.config['beta_start'] = args.diff_beta_start # 0.00001  Starting noise level
        diff_scheduler.config['beta_end'] = args.diff_beta_end # 0.00002  You may need to adjust this based on performance
        diff_scheduler.config['beta_schedule'] = args.diff_beta_scheduler # Consider experimenting with 'cosine' or custom schedules
        diff_scheduler.save_config("diff_scheduler")
        diff_scheduler = DDPMScheduler.from_pretrained("diff_scheduler")
    
    else:
        diff_model = None
        diff_scheduler = None

    ##############################
    # Load training data (and poisons/target)
    ##############################

    train_transforms = get_train_transforms(args)

    if args.poison_mode == 'from_scratch':
        test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar_mean_gm, std=cifar_std_gm)])
    else:
        test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar_mean, std=cifar_std)])

    poison_tuple_list, poison_indices, target_mask_label = get_poisons(args,target_index)
    if poison_tuple_list is None:
        print(f'Error loading poisons for {args.poison_type} for {target_index}, Error: {target_mask_label}')
        xm.rendezvous('training end!')
        return

    train_data_poisoned_ebm = get_base_poisoned_dataset(args,poison_tuple_list, poison_indices, ebm_model, 
                                                        diff_model,diff_scheduler,
                                                        device,
                                                    train_transform=train_transforms)
    
    if args.model in ['HLB','ResNet18_HLB']:
        poisoned_ebm_loader = train_data_poisoned_ebm
        if args.verbose: print(f'IMages Mean, std: {poisoned_ebm_loader.images.mean()}, {poisoned_ebm_loader.images.std()}')
    else:
        poisoned_ebm_loader = torch.utils.data.DataLoader(train_data_poisoned_ebm, batch_size=args.batch_size, shuffle=True,num_workers=4)

    if args.defense == 'Friendly':
        if args.poison_mode == 'from_scratch':
            train_data_poisoned_ebm_noaugs = get_base_poisoned_dataset(args,poison_tuple_list, poison_indices, ebm_model, device,
                                                                train_transform=transforms.Normalize(mean=cifar_mean_gm, std=cifar_std_gm))
        else:
            train_data_poisoned_ebm_noaugs = get_base_poisoned_dataset(args,poison_tuple_list, poison_indices, ebm_model, device,
                                                                    train_transform=transforms.Normalize(mean=cifar_mean, std=cifar_std))
        poisoned_ebm_loader_noaugs = torch.utils.data.DataLoader(train_data_poisoned_ebm_noaugs, batch_size=args.batch_size, shuffle=True,num_workers=4)

    p_count = sum(p.sum().item() for _, _, _, p in poisoned_ebm_loader) 
    if args.model in ['HLB','ResNet18_HLB']:
        if args.verbose: 
            print(f'Loaded training data with {args.poison_type} poison, {p_count} samples , {p_count/len(poisoned_ebm_loader.images):.2%} poisoned, {len(poisoned_ebm_loader.images)} length')
    else:
        if args.verbose: 
            print(f'Loaded training data with {args.poison_type} poison, {p_count} samples , {p_count/len(poisoned_ebm_loader.dataset):.2%} poisoned, {len(poisoned_ebm_loader.dataset)} length')

    
    ##############################
    # Load Test data (and poison target)
    ##############################
            

    if args.model in ['HLB','ResNet18_HLB']:
        test_loader = CifarLoader(None, train=False, batch_size=1000,path=args.data_dir)

    else:

        if args.dataset == 'cifar10':
            # The test set of clean CIFAR10
            if args.device_type == 'xla': 
                if os.path.exists(os.path.join(args.data_dir, 'cifar-10-batches-py')):
                    test_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=test_transforms)
                else: 
                    if xm.is_master_ordinal(): 
                        test_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transforms)
                        xm.rendezvous('download end!')
                    else: 
                        xm.rendezvous('download end!')
                        test_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=test_transforms)
            else:     
                test_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=(not os.path.exists(os.path.join(args.data_dir, 'cifar-10-batches-py'))), transform=test_transforms)
        elif args.dataset == 'cinic10': 
            test_data = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'CINIC-10/test'), transform=test_transforms) 
            cifar_test_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=(not os.path.exists(os.path.join(args.data_dir, 'cifar-10-batches-py'))), transform=test_transforms)
            cifar_test_loader = torch.utils.data.DataLoader(cifar_test_data, batch_size=128,num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=128,num_workers=4)
        
    if args.poison_type == 'Narcissus':
        test_trigger_loaders = get_poisons_target(args, target_index, test_transforms, target_mask = target_mask_label)
    else:
        poison_target_image, target_orig_label = get_poisons_target(args, target_index, test_transforms)

    if args.verbose: print(f'Loaded the test data with poison type {args.poison_type}, length {len(test_loader.images)}')

    ##############################
    # Load the target network, optimizer, and loss function
    ##############################

    # Load the target network
    target_net = load_target_network(args,device)

    if args.verbose: print(f'Loaded target network {args.model} with num params: {sum(p.numel() for p in target_net.parameters())}')

    # Optimizer
    optimizer = get_optimizer(args,target_net)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay, gamma=0.1)

    if args.model in ['HLB','ResNet18_HLB']:
        total_train_steps = np.ceil(len(train_data_poisoned_ebm) * args.epochs)
        lr_schedule = np.interp(np.arange(1+total_train_steps),
                            [0, int(0.2 * total_train_steps), total_train_steps],
                            [0.2, 1, 0])*0.85 # triangular learning rate schedule
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
        if args.verbose: print(f'Loaded the HLB scheduler with {total_train_steps} steps')

        # init_whitening_conv(target_net[0], poisoned_ebm_loader.images)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # define RMD loss functions 
    # z, per_sample_loss, per_sample_criterion, total_loss, lmbda, rmd_criterion = get_RMD_losses(device,poisoned_ebm_loader)
    N = 50_000
    z =  torch.normal(0.0, 0.0000005, size=(N,), device=device)#.type(torch.float64) #small z
    per_sample_criterion = customCrossEntropy(device, reduction = 'none').to(device)#.type(torch.float64)
    lmbda = 0.1
    rmd_criterion = RMD_Loss(per_sample_criterion,device=device, num_datapoints=N) #losses are averaged in minibatch    

    if args.model == 'HLB':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    ##############################
    # Run the attack (Training Loop)
    ##############################

    # Initialize tqdm on master device only
    if (args.device_type == 'xla' and xm.is_master_ordinal()) or args.device_type != 'xla':
        pbar = tqdm(total=args.epochs)

    # Training Logs
    logs = {'train_loss': [], 'test_acc': [], 'p_acc': []}

    if args.defense == 'Epic':
        times_selected = torch.zeros(len(train_data_poisoned_ebm), dtype=torch.int32)
        base_loader = poisoned_ebm_loader
        logs['subset_size'] = {}
    if args.defense == 'EBM' and args.purify_freq > 0: 
        base_loader = poisoned_ebm_loader

    if args.poison_type == 'Narcissus': 
        logs['t_acc'] = []
        
    if args.dataset == 'cinic10':
        logs['cifar_acc'] = []


    train_start_time = time.time()

    target_net.train()

    for epoch in range(args.epochs):

        # _______________________________________________________________________________________
        # Craft Friendly Noise if Friendly Defense
        if args.defense == 'Friendly' and 'friendly' in args.friendly_noise_type and epoch == args.friendly_begin_epoch:
            friendly_noise = generate_friendly_noise(target_net,poisoned_ebm_loader_noaugs,device,args.device_type,friendly_epochs=args.friendly_epochs,mu=args.friendly_mu,
                                                        friendly_lr=args.friendly_lr, clamp_min=-args.friendly_clamp / 255, clamp_max=args.friendly_clamp / 255,model_train=True,
                                                    )
            target_net.zero_grad()
            if args.device_type == 'xla': xm.mark_step()
            if args.verbose: print(f"Friendly noise stats:  Max: {torch.max(friendly_noise)}  Min: {torch.min(friendly_noise)}  Mean (abs): {torch.mean(torch.abs(friendly_noise))}  Mean: {torch.mean(friendly_noise)}")
            train_data_poisoned_ebm.set_perturbations(friendly_noise)
            poisoned_ebm_loader = torch.utils.data.DataLoader(train_data_poisoned_ebm, batch_size=args.batch_size, shuffle=True,num_workers=4)
        # _______________________________________________________________________________________
        # _______________________________________________________________________________________
        # Epic Defense
        if ((args.defense == 'Epic') and (args.epic_subset_size < 1) and (epoch % args.epic_subset_freq == 0) and (epoch >= args.epic_drop_after) and (epoch <= args.epic_stop_after)):

            poisoned_ebm_loader = run_epic(args, target_net, base_loader, epoch, device, times_selected)
            
            if args.verbose: print(f'New training set size: {len(poisoned_ebm_loader.dataset)}')

            logs['subset_size'][epoch] = len(poisoned_ebm_loader.dataset)
        # _______________________________________________________________________________________
        
        # Purify if EBM defense and purify every epoch
        if args.defense == 'EBM' and args.purify_freq > 0 and epoch % args.purify_freq == 0:
            train_data_poisoned_ebm = PoisonedDataset_EBM(args, 
                                                          base_loader, 
                                                          ebm_model, 
                                                          train_transforms, 
                                                          device, 
                                                          n_steps=args.langevin_steps)
            poisoned_ebm_loader = torch.utils.data.DataLoader(train_data_poisoned_ebm, batch_size=args.batch_size, shuffle=True,num_workers=4)
        
        # Train the model
        logs['train_loss'].append(0)
        logs['test_acc'].append(0)
        logs['p_acc'].append(0)
        if args.poison_type == 'Narcissus': logs['t_acc'].append(0) 

        target_net.train()
        # xm.master_print(z)
        for input, target, index, p in poisoned_ebm_loader:
            
            input, target = input.to(device), target.to(device)
            input = Variable(input)
            target = Variable(target)    
            output = target_net(input)
            # xm.master_print(index)
            # Backward pass
            if True:
                # lets do explicit regularization
                
                if args.model in ['HLB','ResNet18_HLB']:
                    # optimizer.zero_grad(set_to_none=True)
                    # output = target_net(input)
                    # sample_loss = per_sample_criterion(output, target)   # per-sample loss in each batch

                    # rmd_criterion.set_z_values(z, index)
                    # rmd_loss = rmd_criterion(output, target)        
        
                    # # gradient clipping 
                    # torch.nn.utils.clip_grad_norm_(target_net.parameters(), 1)
                    # rmd_loss.backward()
                    # optimizer.step()
                    # lr = scheduler.get_last_lr()[0]
                    # # Compute c_i for batch 
                    # c_batch = lr * (z[index] - torch.sqrt(2*sample_loss.clone().detach()))
                    # c_batch = c_batch.clone().detach()

                    # # Update z-auxiliary variables
                    # z[index] -=  (lmbda * c_batch)      
                    
                    optimizer.zero_grad(set_to_none=True)
                    def closure():
                        output = target_net(input)
                        sample_loss = per_sample_criterion(output, target)   # per-sample loss in each batch
                        
                        rmd_criterion.set_z_values(z, index)
                        rmd_loss = rmd_criterion(output, target)   
                        torch.nn.utils.clip_grad_norm_(target_net.parameters(), 1)
                        rmd_loss.backward()     
                        return rmd_loss, output,sample_loss
                    # gradient clipping 
                    # torch.nn.utils.clip_grad_norm_(target_net.parameters(), 1)
                    # rmd_loss, output,sample_loss = rmd_loss.backward(closure)
                    rmd_loss, output,sample_loss = optimizer.step(closure)
                    lr = scheduler.get_last_lr()[0]
                    # Compute c_i for batch 
                    c_batch = lr * (z[index] - torch.sqrt(2*sample_loss.clone().detach()))
                    c_batch = c_batch.clone().detach()

                    # Update z-auxiliary variables
                    z[index] -=  (lmbda * c_batch)      
                    scheduler.step()                      
                    
                    logs['train_loss'][-1] += sample_loss.mean().item()
                                   
                else:
                    optimizer.zero_grad()
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    # Update logs
                    logs['train_loss'][-1] += loss.item()
                
            else:
                if args.model in ['HLB','ResNet18_HLB']:
                    loss = F.cross_entropy(output, target, reduction='none')
                    logs['train_loss'][-1] += loss.mean().item()
                    # train_acc.append((outputs.detach().argmax(1) == labels).float().mean().item())
                    optimizer.zero_grad(set_to_none=True)
                    loss.sum().backward()
                    optimizer.step()
                    scheduler.step()
                else:
                    optimizer.zero_grad()
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    # Update logs
                    logs['train_loss'][-1] += loss.item()

            if args.device_type == 'xla': xm.mark_step()

            if args.defense == 'Epic':
                times_selected[index] += 1

        # Update logs
        logs['train_loss'][-1] /= len(poisoned_ebm_loader)
        
        # Decay the learning rate
        if args.model not in ['HLB','ResNet18_HLB']: scheduler.step()
        if args.device_type == 'xla': xm.mark_step()

        # Test the model for from_scratch attacks
        if args.poison_mode == 'from_scratch':
            if args.model in ['HLB','ResNet18_HLB']:
                test_acc = eval_HLB(target_net, test_loader, device)
            else:
                test_acc = get_test_acc(target_net, test_loader, device)
            logs['test_acc'][-1] = test_acc
            
            if args.dataset == 'cinic10':
                cifar_acc = get_test_acc(target_net, cifar_test_loader, device)
                logs['cifar_acc'].append(cifar_acc)

        if args.poison_type == 'Narcissus' and args.poison_mode == 'from_scratch':
            _, p_acc, t_acc = run_test_epoch_narcissus(test_trigger_loaders[1], target_net, nn.CrossEntropyLoss(reduction='none'),target_index, device)
            logs['p_acc'][-1] = p_acc
            logs['t_acc'][-1] = t_acc
        elif args.poison_mode == 'transfer' and args.poison_type != 'Narcissus':
            target_pred = target_net(poison_target_image.to(device).view(1,3,32,32))
            pred = torch.argmax(target_pred).item()
            success = bool(pred == target_mask_label)
            logs['p_acc'][-1] = success

        # Update progress bar
        if (args.device_type == 'xla' and xm.is_master_ordinal()) or args.device_type != 'xla':
            pbar.update(1)
            if args.poison_type == 'Gradient_Matching':
                pbar.set_description(f'Epoch {epoch+1}/{args.epochs} | Test Acc {logs["test_acc"][-1]:.4f} | Poison Success {logs["p_acc"][-1]} | ')
            elif args.poison_type == 'Narcissus':
                pbar.set_description(f'Epoch {epoch+1}/{args.epochs} | T Acc {logs["test_acc"][-1]:.4f} | P Acc {logs["p_acc"][-1]:.4f} | ')
            else:
                pbar.set_description(f'Index {target_index} | Poison Type {args.poison_type} | Defense {args.defense} | Epoch {epoch+1}/{args.epochs}')

    training_time = time.time() - train_start_time

    # Get the final test accuracy and poison success
    end_acc = get_test_acc(target_net, test_loader, device)

    if args.poison_type != 'Narcissus':
        target_pred = target_net(poison_target_image.to(device).view(1,3,32,32))
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
    
    if args.poison_type != 'Narcissus':
        get_accs_save_results(args, rank, target_index, end_acc, success, correct_class, training_time, logs)
    else:
        get_accs_save_results_Narcissus(args, rank, target_index, end_acc, training_time, logs, p_accs, t_accs)

    # Save the model
    if args.save_models:
        model_save_dir = os.path.join(args.output_dir,'Models',f'{args.defense}',f'{args.experiment_timestamp}_Index_{target_index}')
        if os.path.exists(model_save_dir) == False: os.makedirs(model_save_dir)
        torch.save(target_net.to('cpu').state_dict(), os.path.join(model_save_dir,'model.pt'))

    # Print the results
    if args.poison_type != 'Narcissus':
        print(f'Index {target_index} | Poison Type {args.poison_type} | Defense {args.defense} | End Acc {end_acc} | Poison Success {success} | Correct Pred {correct_class} | Training Time {training_time}')
    elif args.poison_type == 'Narcissus':
        if args.dataset == 'cinic10':
            print(f'Index {target_index} | Poison Type {args.poison_type} | Defense {args.defense} | P1 Acc {p_accs[1]:.2%} | T1 Acc {t_accs[1]:.2%} | End Acc {end_acc:.3f}% | CIFAR Acc {cifar_acc:.3f}% | Training Time {training_time:.1f} ')
        else: 
            print(f'Index {target_index} | Poison Type {args.poison_type} | Defense {args.defense} | P1 Acc {p_accs[1]:.2%} | T1 Acc {t_accs[1]:.2%} | End Acc {end_acc:.3f}% | Training Time {training_time:.1f}')
    
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
    parser.add_argument('--verbose','--v', default=False, action='store_true',help='print out additional information when running')
    parser.add_argument('--seed', default=11, type=int,help='seed for reproducibility')
    parser.add_argument('--save_models', default=False, action='store_true',help="Whether to save the models")
    parser.add_argument('--exp_name', default=None, type=str,help='name of the experiment to append to the output dataframe')
    parser.add_argument('--no_poison', default=False, action='store_true',help='whether to run the attack or not')

    ### Experiment Arguments ###
    parser.add_argument('--poison_mode', default='from_scratch', type=str, choices=['from_scratch','transfer'],help='mode of attack')
    parser.add_argument('--poison_type', default='Narcissus', type=str, choices=['Narcissus', 'Gradient_Matching','BullseyePolytope','BullseyePolytope_Bench'],help='type of poison to generate')
    parser.add_argument('--fine_tune', default=False, action='store_true',help="Whether retrain the full model (fine-tuning) or just the linear layer (default: False)")
    parser.add_argument('--defense', default='EBM', type=str, choices=['None','EBM','Epic','Friendly','Diff','EBM_Diff'],help='type of defense to use')
    parser.add_argument('--start_target_index', default=0, type=int,help='start label for the attack (only used for from_scratch attacks)')
    parser.add_argument('--selected_indices', default=None, nargs='+', type=int, help='Specific indices to run the attack on each TPU core (default: None, TPU only!!!)')
    parser.add_argument('--model', default='ResNet18', type=str, choices=['ResNet18','ResNet18_HLB','HLB','MobileNetV2','DenseNet121'],help='type of model to use')

    ### HLB Arguments ###
    parser.add_argument('--hlb_flip', default=True, action='store_false',help='whether to use flip in HLB')
    parser.add_argument('--hlb_translate', default=4, type=int,help='whether to use translate in HLB')
    parser.add_argument('--hlb_cutout', default=None, type=int,help='whether to use cutout in HLB')

    ### EBM Arguments ###
    args_ebm = parser.add_argument_group('EBM')
    args_ebm.add_argument('--ebm', default=None, type=str,  help='path to the ebm model')
    args_ebm.add_argument('--ebm_nf', default=128, type=int,  help='number of filters for the ebm model')
    args_ebm.add_argument('--langevin_steps', default=150, type=int, help='number of langevin steps')
    args_ebm.add_argument('--langevin_temp', default=1e-4, type=float, help='langevin temperature')
    args_ebm.add_argument('--purify_reps', default=1, type=int,  help='number of purification repetitions')
    args_ebm.add_argument('--purify_reps_mode', default='repeat', type=str, choices=['repeat','mean','median'],help='how to aggregate the purification repetitions')
    args_ebm.add_argument('--langevin_eps', default=1.25e-2, type=float,  help='langevin eps')
    args_ebm.add_argument('--ebm_perturb_clamp', default=None, type=int,help='clamp for EBM perturbation')
    args_ebm.add_argument('--purify_freq', default=0, type=int, help="Whether to use the penultimate features for training (default: False)")
    args_ebm.add_argument('--pre_purify_steps', default=100, type=int, help='number of pre purification steps only if purify_freq > 0')

    ### Diffusion Arguments ###
    args_diff = parser.add_argument_group('Diffusion')
    args_diff.add_argument('--diff_steps', default=125, type=int, help='number of diffusion steps')
    args_diff.add_argument('--diff_beta_start', default=0.00001, type=float, help='starting noise level')
    args_diff.add_argument('--diff_beta_end', default=0.00002, type=float, help='ending noise level')
    args_diff.add_argument('--diff_beta_scheduler', default='linear', type=str, choices=['linear','cosine'], help='diffusion beta scheduler')

    ### Other Defense and Poison Arguments ###
    parser.add_argument('--iters_bp', default=800, type=int,help='iterations for making poison')
    parser.add_argument('--num_images_bp', default=50, type=int,help='number of poisoned images generated')
    parser.add_argument('--net_repeat_bp', default=1, type=int, help='number of times to repeat the network for methods BP-1, BP-3, BP-5')
    parser.add_argument('--num_per_class_bp', default=50, type=int, help='num of samples per class for re-training, or the poison dataset')
    parser.add_argument('--noise_sz_narcissus', default=32, type=int, help='size of the noise trigger for Narcissus')
    parser.add_argument('--noise_eps_narcissus', default=8, type=int, help='epsilon for the noise trigger for Narcissus')
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

    if args.poison_type == 'Gradient_Matching' and args.poison_mode == 'transfer':
        raise ValueError('Gradient Matching does not support transfer attacks')
    # if args.poison_type == 'Narcissus' and args.poison_mode == 'transfer':
    #     raise ValueError('Narcissus does not support transfer attacks')
    if args.poison_type == 'BullseyePolytope_Bench' and args.poison_mode == 'from_scratch':
        raise ValueError('BullseyePolytope_Bench does not support from_scratch attacks')
    if args.poison_type == 'BullseyePolytope' and args.poison_mode == 'from_scratch':
        raise ValueError('BullseyePolytope does not support from_scratch attacks')
    if args.selected_indices is not None and args.device_type != 'xla':
        raise ValueError('selected_indices only supported for TPU')
    if args.selected_indices is not None and args.poison_mode != 'from_scratch':
        raise ValueError('selected_indices only supported for from_scratch attacks')

    ##############
    # Directories
    ##############

    # Setup directories for remote server
    if args.remote_user is not None:
        args.data_dir = args.data_dir.replace('/home',f'/home/{args.remote_user}')
        args.output_dir = args.output_dir.replace('/home',f'/home/{args.remote_user}')
        args.models_dir = args.models_dir.replace('/home',f'/home/{args.remote_user}')

    # Create the output directory and get the timestamp
    args.experiment_timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
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
