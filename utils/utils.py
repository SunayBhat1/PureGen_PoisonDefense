import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import json
import glob


from models.Classifiers import load_model
from utils.classifier import *

try: import torch_xla.core.xla_model as xm
except: pass

dataset_dict = {'cifar10':{'num_classes':10,'img_dim':32},
                'cinic10':{'num_classes':10,'img_dim':32},
                'tinyimagenet':{'num_classes':200,'img_dim':64},
                'stl10':{'num_classes':10,'img_dim':96},
                'stl10_64':{'num_classes':10,'img_dim':64},
                }

######################
# Argument Functions #
######################
                
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

def save_purify_time(data_key, purify_time,args, save_path, file_name = 'PurifyTimes.csv'):
    # Check if the directory exists, create it if it doesn't
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, file_name)

    # Check if the file exists
    if os.path.isfile(file_path):
        # If the file exists, load the existing data
        df = pd.read_csv(file_path)
    else:
        # If the file doesn't exist, create a new dataframe
        df = pd.DataFrame(columns=['Data Key','Dataset','Purify Time','Args'])

    # Append results to the dataframe
    df = pd.concat([df, pd.DataFrame({'Data Key': data_key, 
                                      'Dataset': args.dataset,
                                      'Purify Time': purify_time,
                                      'Args': json.dumps(vars(args))}, 
                                      index=[0])], ignore_index=True)
    
    # Save the dataframe
    df.to_csv(file_path, index=False)

def check_training_end(args,target_index):
    if args.poison_type == 'Narcissus' and target_index >= 10: 
        return True
    elif args.poison_type == 'GradientMatching' and target_index >= 100:
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


####################
# Eval Functions   #
####################

def eval_epoch(args,target_net, logs, test_loader, device, test_trigger_loaders=None, poison_target_image=None, target_mask_label=None, target_index=None):

    test_acc = get_test_acc(target_net, test_loader, device)
    logs['test_acc'].append(test_acc)
        
    if args.poison_type != 'NeuralTangent':
        if args.poison_type == 'Narcissus':
            _, p_acc, t_acc = run_test_epoch_narcissus(test_trigger_loaders[1], target_net, nn.CrossEntropyLoss(reduction='none'),target_index, device)
            logs['p_acc'].append(p_acc)
            logs['t_acc'].append(t_acc)

        else:
            img_dim = dataset_dict[args.dataset]['img_dim']
            target_pred = target_net(poison_target_image.to(device).view(1,3,img_dim,img_dim))
            pred = torch.argmax(target_pred).item()
            success = bool(pred == target_mask_label)
            logs['p_acc'].append(success)

    return logs

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

    acc = correct / total
    return acc

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

##################
# Pbar Functions #
##################

def update_progress_bar(args, pbar, epoch, logs):
    """
    This function updates the progress bar with the test accuracy and poison success rate.

    Parameters:
        args: The command-line arguments.
        pbar: The progress bar.
        epoch (int): The current epoch.
        logs (dict): The logs dictionary containing 'test_acc' and 'p_acc'.
    """

    # Update progress bar
    pbar.update(1)
    if args.poison_mode == 'clean' or args.poison_type == 'NeuralTangent':
        pbar.set_description(f'Epoch {epoch+1}/{args.epochs} | Test Acc {logs["test_acc"][-1]:.2%}')
    elif args.poison_type != 'Narcissus':
        pbar.set_description(f'Epoch {epoch+1}/{args.epochs} | Test Acc {logs["test_acc"][-1]:.2%} | Poison Success {logs["p_acc"][-1]} | ')
    elif args.poison_type == 'Narcissus':
        pbar.set_description(f'Epoch {epoch+1}/{args.epochs} | Test Acc {logs["test_acc"][-1]:.2%} | P Acc {logs["p_acc"][-1]:.2%} | ')

##################
# Save Functions #
##################
        
def get_accs_save_results(args, rank, target_index, end_acc, training_time, logs):

    if args.device_type == 'xla':
        df_path = os.path.join(args.output_dir,f'Results_{rank}.csv')
    else:
        df_path = os.path.join(args.output_dir,'Results.csv')

    df = pd.DataFrame(columns=['Model','Dataset','Data Key','Target Index','End Acc','Exp Name','Calc Time','Args','Logs','Train Time'])

    # Convert args to a dictionary and then a json string
    args_dict = vars(args)
    for key, value in args_dict.items():
        if isinstance(value, torch.device):
            args_dict[key] = str(value)
    args_str = json.dumps(args_dict)

    # Convert the logs to a json string
    logs_str = json.dumps(logs)

    # Append results to the dataframe
    df = pd.concat([df, pd.DataFrame({'Model': args.model, 'Dataset': args.dataset,
                                        'Data Key': args.data_key,
                                        'Target Index': target_index,
                                        'End Acc': end_acc,
                                        'Exp Name': args.exp_name,
                                        'Calc Time': args.experiment_timestamp,
                                            'Args': args_str, 'Logs': logs_str, 'Train Time': training_time
                                    }, index=[0])], ignore_index=True)
    # Save the dataframe
    df.to_csv(df_path, index=False)

def get_accs_save_results_untriggered(args, rank, target_index, end_acc, success, correct_class, training_time, logs):

    if args.device_type == 'xla':
        df_path = os.path.join(args.output_dir,f'Results_{rank}.csv')
    else:
        df_path = os.path.join(args.output_dir,'Results.csv')

    df = pd.DataFrame(columns=['Data Key','Model','Dataset','Target Index','End Acc','Success','Correct Pred','Exp Name','Calc Time','Args','Logs','Train Time'])

    # Convert args to a dictionary and then a json string
    args_dict = vars(args)
    for key, value in args_dict.items():
        if isinstance(value, torch.device):
            args_dict[key] = str(value)
    args_str = json.dumps(args_dict)

    # Convert the logs to a json string
    logs_str = json.dumps(logs)

    # Append results to the dataframe
    df = pd.concat([df, pd.DataFrame({'Data Key': args.data_key, 'Model': args.model, 'Dataset': args.dataset,
                                        'Target Index': target_index, 
                                        'End Acc': end_acc,
                                        'Success': success, 'Correct Pred': correct_class,
                                        'Exp Name': args.exp_name,
                                        'Calc Time': args.experiment_timestamp,
                                        'Args': args_str, 'Logs': logs_str, 'Train Time': training_time
                                    }, index=[0])], ignore_index=True)
    
    # Save the dataframe
    df.to_csv(df_path, index=False)
    
def get_accs_save_results_triggered(args, rank, target_index, end_acc, training_time, logs, p_accs, t_accs):

    if args.device_type == 'xla':
        df_path = os.path.join(args.output_dir,f'Results_{rank}.csv')
    else:
        df_path = os.path.join(args.output_dir,'Results.csv')

    df = pd.DataFrame(columns=['Data Key','Model','Dataset','Target Index','End Acc',
                                'P1 Acc','T1 Acc','Exp Name',
                                'Calc Time','Train Time',
                                'P2 Acc','T2 Acc',
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

    df = pd.concat([df, pd.DataFrame({'Data Key': args.data_key, 'Model': args.model, 'Dataset': args.dataset,
                                        'Target Index': target_index, 
                                        'End Acc': end_acc,
                                        'P1 Acc': p_accs[1], 'T1 Acc': t_accs[1],
                                        'Exp Name': args.exp_name,
                                        'Calc Time': args.experiment_timestamp,
                                        'Train Time': training_time,
                                        'P2 Acc': p_accs[2], 'T2 Acc': t_accs[2],
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

##################
# Argparse Import Helper Functions
##################

def int_or_int_list(s):
    if ',' in s:
        try:
            return [int(item) for item in s.split(',')]
        except ValueError:
            return int(s)
    else:
        return int(s)
    
def float_or_float_list(s):
    if ',' in s:
        try:
            return [float(item) for item in s.split(',')]
        except ValueError:
            return float(s)
    else:
        return float(s)

def str_or_str_list(s):
    if ',' in s:
        return s.split(',')
    else:
        return s

def none_or_str(value):
    if value == 'None':
        return None
    return value