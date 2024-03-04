import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import json
import glob


from models import load_model
from utils.utils_data import *
from utils.utils_optim import *
try: import torch_xla.core.xla_model as xm
except: pass

def set_args_from_config(args, config, section_name):

    for key, value in config[section_name].items():

        if value.isdigit(): 
            value = int(value)
        elif ',' in value:
            # Convert comma-separated string to list of integers
            value = list(map(int, value.split(',')))
        else:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == 'true': 
                    value = True
                elif value.lower() == 'false': 
                    value = False

        setattr(args, key, value)

def set_seed(seed,device,device_type='xla'):

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device_type == 'xla':
            import torch_xla.core.xla_model as xm
            xm.set_rng_state(seed,device=device)
        elif device_type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

def get_device(device_type='xla'):
    if device_type == 'xla':
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    elif device_type == 'cuda':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_type == 'cpu':
        return torch.device("cpu")
    else:
        raise ValueError('device_type must be xla or cuda or cpu')
    

def check_training_end(args,target_index):
    if args.poison_type == 'Narcissus' and target_index >= 10: 
        return True
    elif args.poison_type == 'Gradient_Matching' and target_index >= 100:
        return True
    elif args.poison_type == 'BullseyePolytope' and target_index >= 50:
        return True
    elif args.poison_type == 'BullseyePolytope_Bench' and target_index >= 100:
        return True
    return False

def set_target_index_and_check_end(args, rank):
    # Set the poison target index (image or class label)
    if args.selected_indices is None:
        target_index = rank + args.start_target_index
    else:
        try:
            target_index = args.selected_indices[rank]
        except:
            return None

    if args.device_type == 'xla':
        if check_training_end(args,target_index):
            return None

    return target_index

def load_target_network(args,device):

    if args.poison_mode == 'from_scratch' or args.fine_tune:
        target_net = load_model(args.model)
    elif args.poison_type == 'BullseyePolytope':
        target_net = load_model(args.model, eval_bn=True)
    elif args.poison_type == 'BullseyePolytope_Bench':
        target_net = load_model(args.model, num_classes=100, eval_bn=True)
    else:
        target_net = load_model(args.model)

    # Load state dict for transfer learning
    if args.poison_mode == 'transfer':
            
        # Use benchmark if running BullseyePolytope_Bench
        if args.poison_type == 'BullseyePolytope_Bench':
            model_path = 'ResNet18_CIFAR100.pth'
        else:
            model_path = args.model_path

        state_dict_path = os.path.join(args.data_dir,'models', 'transfer_models', model_path)
        state_dict_module = torch.load(state_dict_path,map_location=torch.device('cpu'))['net']
        state_dict = {}
        for k,v in state_dict_module.items():
            state_dict[k.replace('module.', '')] = v
        target_net.load_state_dict(state_dict)

        if args.verbose: print(f'Loaded the target network from {state_dict_path}')

    # Move target_net to device
    if args.model in ['HLB','ResNet18_HLB']:
        target_net = target_net.to(device).to(memory_format=torch.channels_last)
    else:
        target_net = target_net.to(device)

    # Reinit Linear layer for transfer learning
    if args.poison_mode == 'transfer' and args.reinit_linear:
        if args.model == 'ResNet18':
            target_net.linear = nn.Linear(512, 10).to(device)
        elif args.model == 'DenseNet121':
            target_net.linear = nn.Linear(1024, 10).to(device)
        elif args.model == 'MobileNetV2':
            target_net.linear = nn.Linear(1280, 10).to(device)
        if args.verbose: print(f'Reinitialized the linear layer of the target network')

    return target_net

def get_optimizer(args,target_net):

    if args. poison_mode == 'from_scratch':
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(target_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'sgd':
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in target_net.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in target_net.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = torch.optim.SGD(grouped_parameters, lr=args.lr, momentum=args.momentum, nesterov=True)
        elif args.optim == 'smd':
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in target_net.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in target_net.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = SMD_qnorm(grouped_parameters, lr=args.lr, momentum=args.momentum, nesterov=True,q=1.25)
        elif  args.optim == 'adamwq':
            optimizer = AdamWq(target_net.parameters(), lr=args.lr, weight_decay=args.weight_decay,q=1.25)
            
    elif args.poison_mode == 'transfer':

        if args.fine_tune:
            params = target_net.parameters()
        else:
            params = target_net.get_penultimate_params_list()

        if args.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # if args.model in ['HLB','ResNet18_HLB']:

    #     optimizer = torch.optim.SGD(target_net.parameters(), lr=args.lr/args.batch_size, momentum=args.momentum, nesterov=True,
    #                             weight_decay=args.weight_decay*args.batch_size)
    #     # kilostep_scale = 1024 * (1 + 1 / (1 - args.momentum))
    #     # lr = args.lr / kilostep_scale # un-decoupled learning rate for PyTorch SGD
    #     # wd = args.weight_decay * args.batch_size / kilostep_scale
    #     # lr_biases = lr * args.bias_scaler

    #     # print(f'lr: {lr}, lr_biases: {lr_biases}, wd: {wd}')

    #     # norm_biases = [p for k, p in target_net.named_parameters() if 'norm' in k and p.requires_grad]
    #     # other_params = [p for k, p in target_net.named_parameters() if 'norm' not in k and p.requires_grad]
    #     # param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
    #     #              dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    #     # optimizer = torch.optim.SGD(param_configs, momentum=args.momentum, nesterov=True)

    return optimizer


def get_test_acc(net, loader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc

def eval_HLB(model, loader,device):
    model.eval()
    with torch.no_grad():
        outs = []
        # Iterate over each batch from the loader
        for inputs, _ in loader:
            inputs = inputs.to(device)
            # Apply the model to the inputs and their flipped versions, then sum the results
            output = model(inputs) + model(inputs.flip(-1))
            # Append the output to the list of outputs
            outs.append(output)
            xm.mark_step()

        # Concatenate the list of outputs into a single tensor
        outs = torch.cat(outs)
    return (outs.argmax(1) == loader.labels.to(device)).float().mean().item()

def run_test_epoch_narcissus(test_loader, model, loss_fn, poisoned_label, device):
    model.eval()

    losses = []
    p_acc = []
    t_acc = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            model.eval()

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets).mean()
            pred = outputs.max(1)[1].cpu().detach().numpy()

            losses.append(loss.item())
            p_acc.append((pred == poisoned_label).mean())
            t_acc.append((pred == targets.cpu().detach().numpy()).mean())
            
        losses = np.mean(losses)
        p_acc = np.mean(p_acc)
        t_acc = np.mean(t_acc)
    
    return losses, p_acc, t_acc


def get_accs_save_results(args, rank, target_index, end_acc, success, correct_class, training_time, logs):

    if args.device_type == 'xla':
        df_path = os.path.join(args.output_dir,f'Results_{rank}.csv')
    else:
        df_path = os.path.join(args.output_dir,'Results.csv')

    df = pd.DataFrame(columns=['Defense','Target Index','End Acc','Success','Correct Pred','Exp Name','Calc Time','Args','Logs','Train Time'])

    # Convert args to a dictionary and then a json string
    args_dict = vars(args)
    for key, value in args_dict.items():
        if isinstance(value, torch.device):
            args_dict[key] = str(value)
    args_str = json.dumps(args_dict)

    # Convert the logs to a json string
    logs_str = json.dumps(logs)

    # Append results to the dataframe
    df = pd.concat([df, pd.DataFrame({'Defense': args.defense, 'Target Index': target_index, 
                                            'End Acc': end_acc,
                                            'Success': success, 'Correct Pred': correct_class,
                                            'Exp Name': args.exp_name,
                                            'Calc Time': args.experiment_timestamp,
                                            'Args': args_str, 'Logs': logs_str, 'Train Time': training_time
                                            }, index=[0])], ignore_index=True)
    
    # Save the dataframe
    df.to_csv(df_path, index=False)
    
def get_accs_save_results_Narcissus(args, rank, target_index, end_acc, training_time, logs, p_accs, t_accs):

    if args.device_type == 'xla':
        df_path = os.path.join(args.output_dir,f'Results_{rank}.csv')
    else:
        df_path = os.path.join(args.output_dir,'Results.csv')

    df = pd.DataFrame(columns=['Defense','Target Index','End Acc',
                                'P1 Acc','T1 Acc','Exp Name',
                                'Calc Time','Train Time',
                                'P2 Acc','T2 Acc', 'P3 Acc','T3 Acc',
                                'Args','Logs'
                                ])
    
    # Convert args to a dictionary and then a json string
    args_dict = vars(args)
    for key, value in args_dict.items():
        if isinstance(value, torch.device):
            args_dict[key] = str(value)
    args_str = json.dumps(args_dict)

    # Convert the logs to a json string
    logs_str = json.dumps(logs)

    df = pd.concat([df, pd.DataFrame({'Defense': args.defense, 'Target Index': target_index, 
                                            'End Acc': end_acc,
                                            'P1 Acc': p_accs[1], 'T1 Acc': t_accs[1],
                                            'Exp Name': args.exp_name,
                                            'Calc Time': args.experiment_timestamp,
                                            'Train Time': training_time,
                                            'P2 Acc': p_accs[2], 'T2 Acc': t_accs[2],
                                            'P3 Acc': p_accs[3], 'T3 Acc': t_accs[3],
                                            'Args': args_str, 'Logs': logs_str, 
                                            }, index=[0])], ignore_index=True)
    
    # Save the dataframe
    df.to_csv(df_path, index=False)


def concat_result_dataframes_xla(args):
    
    df_paths = glob.glob(os.path.join(args.output_dir, 'Results_*.csv'))

    df_list = [pd.read_csv(df_path) for df_path in df_paths]
    df = pd.concat(df_list, ignore_index=True)

    if os.path.exists(os.path.join(args.output_dir,f'Results.csv')):
        df_results = pd.read_csv(os.path.join(args.output_dir,f'Results.csv'))
        df = pd.concat([df_results, df], ignore_index=True)
        df.to_csv(os.path.join(args.output_dir,f'Results.csv'), index=False)
    else:
        df.to_csv(os.path.join(args.output_dir,f'Results.csv'), index=False)
    # Delete the individual dataframes
    for df_path in df_paths:
        os.remove(df_path)

