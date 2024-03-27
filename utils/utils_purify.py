import torch
from torch import nn
import numpy as np
import time
import os
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from tqdm import tqdm
import pickle

import torch.nn.functional as F
import random

try: import torch_xla.core.xla_model as xm
except: pass

# Used for denormalizing poisons
cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2023, 0.1994, 0.2010)


class ImageListDataset(data.Dataset):
    def __init__(self, image_label_list):
        self.image_label_list = image_label_list

    def __len__(self):
        return len(self.image_label_list)

    def __getitem__(self, idx):
        image, label = self.image_label_list[idx]
        return transforms.ToTensor()(image), label
    

def save_poisons(args,poison_tuple_list, poison_indices, target, data_key):
    """
    This function saves the poison data to a file. The directory is created if it doesn't exist.

    Parameters:
        args (object): An object containing various parameters.
        poison_tuple_list (list): A list of tuples containing the poison data.
        poison_indices (list): A list of indices of the poison data.
        target (object): The target of the poison attack.
        data_key (str): The key used to save the data.

    Returns:
        str: The path to the saved file.
    """

    subfolder = os.path.join(args.data_dir,'PureGen_PoisonDefense',args.dataset,'Poisons')
    # Create the directory if it doesn't exist
    if args.poison_type == 'GradientMatching':
        save_dir = os.path.join(args.data_dir, subfolder, 'GradientMatching')
    elif args.poison_type == 'Narcissus':
        save_dir = os.path.join(args.data_dir, subfolder, f'Narcissus/size={args.noise_sz_narcissus}_eps={args.noise_eps_narcissus}_num={args.num_images_narcissus}')
    elif args.poison_type == 'BullseyePolytope':
        if args.fine_tune: bp_subpath = 'end2end-training'
        else: bp_subpath = 'linear-transfer-learning'
        if args.num_images_bp == 5: bp_subpath = os.path.join(bp_subpath, f'mean-{args.net_repeat_bp}Repeat')
        else: bp_subpath = os.path.join(bp_subpath, f'mean')
        save_dir = os.path.join(args.data_dir, subfolder, f'Bullseye_Polytope/{args.num_images_bp}-imgs/{bp_subpath}/{args.iters_bp}-iters')
    elif args.poison_type == 'BullseyePolytope_Bench':
        save_dir = os.path.join(args.data_dir, subfolder, f'Transfer_Bench/bp_poisons/{args.num_images_bp}-imgs')

    if not os.path.exists(os.path.join(save_dir, data_key)):
        os.makedirs(os.path.join(save_dir, data_key))

    # Save the poison_tuple_list, poison_indices, and target
    torch.save((poison_tuple_list, poison_indices, target), os.path.join(save_dir, data_key, f'{args.target_index}.pth'))

    return os.path.join(save_dir, data_key, f'{args.target_index}.pth')

def get_poisons(args,target_index):
    """
    This function gets the poison data from a file.

    Parameters:
        args (object): An object containing various parameters.
        target_index (int): The index of the target class.

    Returns:
        tuple: A tuple containing the poison data, the indices of the poison data, and the target of the poison attack.
    """
    
    if args.poison_type == 'GradientMatching':
        poison_tuple_list, poison_indices, target = get_poisoned_subset_GM(os.path.join(args.data_dir,f'Poisons/GradientMatching/',args.dataset,'ResNet34_250',f'{target_index}'))

    elif args.poison_type == 'Narcissus':
        if args.poison_mode == 'from_scratch':
            if hasattr(args, 'index_list_narcissus'):
                index_list = np.load(os.path.join(args.data_dir,'models/ebms',f'{args.index_list_narcissus}'))
            else:
                index_list = None
            if args.dataset == 'stl10':
                # Map stl classes to cifar classes
                stl_cifar_label_map = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 7, 7: 6, 8: 8, 9: 9}
                poison_tuple_list, poison_indices, target = get_poisoned_subset_narcissus(os.path.join(args.data_dir,f'Poisons/Narcissus/{args.dataset}/size={args.noise_sz_narcissus}_eps={args.noise_eps_narcissus}/best_noise_lab{stl_cifar_label_map[target_index]}.npy'), 
                                                                                        args.data_dir, args.dataset, target_index, args.num_images_narcissus, not args.random_imgs_narcissus, index_list)
            else:
                poison_tuple_list, poison_indices, target = get_poisoned_subset_narcissus(os.path.join(args.data_dir,f'Poisons/Narcissus/{args.dataset}/size={args.noise_sz_narcissus}_eps={args.noise_eps_narcissus}/best_noise_lab{target_index}.npy'), 
                                                                                        args.data_dir, args.dataset, target_index, args.num_images_narcissus, not args.random_imgs_narcissus, index_list)
        elif args.poison_mode == 'transfer':
            poison_tuple_list, poison_indices, target = get_poisoned_subset_narcissus(os.path.join(args.data_dir,f'Poisons/Narcissus/{args.dataset}/size={args.noise_sz_narcissus}_eps={args.noise_eps_narcissus}/best_noise_lab{target_index}.npy'), 
                                                                                        os.path.join(args.data_dir,'CIFAR10_TRAIN_Split.pth'), args.dataset, target_index, args.num_images_narcissus, not args.random_imgs_narcissus, index_list=None, transfer_subset=True)
    elif args.poison_type == 'BullseyePolytope':

        inverse_transform = transforms.Compose([transforms.Normalize(mean=[-i/j for i,j in zip(cifar_mean, cifar_std)], std=[1/j for j in cifar_std]),transforms.ToPILImage()])

        # Path Adjustments for Bullseye Polytope Settings
        if args.fine_tune: bp_subpath = 'end2end-training'
        else: bp_subpath = 'linear-transfer-learning'

        if args.num_images_bp == 5: bp_subpath = os.path.join(bp_subpath, f'mean-{args.net_repeat_bp}Repeat')
        else: bp_subpath = os.path.join(bp_subpath, f'mean')

        # Load the poison
        bp_poison = torch.load(os.path.join(args.data_dir, \
                        f'Poisons/Bullseye_Polytope/attack-results-{args.num_images_bp}poisons/100-overlap/{bp_subpath}/{args.iters_bp}/{target_index}/poison_{args.iters_bp-1:05d}.pth'),map_location=torch.device('cpu'))
        
        poison_tuple_list, poison_indices = bp_poison['poison'], bp_poison['idx']

        # Unnormalize the poisons
        for i in range(len(poison_tuple_list)):
            poison_tuple_list[i] = (inverse_transform(poison_tuple_list[i][0]), poison_tuple_list[i][1])

        target = 8

    elif args.poison_type == 'BullseyePolytope_Bench':
            
        # Load the poison
        with open(os.path.join(args.data_dir,f'Poisons/Transfer_Bench/bp_poisons/num_poisons={args.num_images_bp}/{target_index}', 'poisons.pickle'), "rb") as handle: 
            poison_tuple_list = pickle.load(handle)
        with open(os.path.join(args.data_dir,f'Poisons/Transfer_Bench/bp_poisons/num_poisons={args.num_images_bp}/{target_index}', 'base_indices.pickle'), "rb") as handle: 
            poison_indices = pickle.load(handle)

        target = poison_tuple_list[0][1]

    return poison_tuple_list, poison_indices, target


def get_poisoned_subset_narcissus(poisons_path, data_dir, dataset, label, poison_amount, last_n=True, index_list=None, transfer_subset=False):
    """
    This function generates a subset of poisoned data for the Narcissus attack.

    Parameters:
        poisons_path (str): The path to the file containing the noise to apply.
        data_dir (str): The directory containing the base dataset.
        dataset (str): The name of the dataset ('cifar10' or 'cinic10').
        label (int): The label of the data to poison.
        poison_amount (int): The number of data points to poison.
        last_n (bool, optional): Whether to poison the last `poison_amount` data points. Default is True.
        index_list (list, optional): A list of indices of the data to poison. If provided, `last_n` is ignored. Default is None.
        transfer_subset (bool, optional): Whether to use a transfer subset. Default is False.

    Returns:
        list: A list of tuples containing the poisoned data and the corresponding labels.
        numpy.ndarray: An array of indices of the poisoned data.
        numpy.ndarray: The noise applied to the data.
    """

    if transfer_subset:
        base_dataset = torch.load(data_dir)['others']
        train_labels = np.array([label for _, label in base_dataset])
    else:
        if dataset == 'cifar10':
            base_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=(not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py'))))
            train_labels = np.array(base_dataset.targets)
        elif dataset == 'cinic10':
            base_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'CINIC-10/valid'))
            train_labels = np.array(base_dataset.targets)
        elif dataset in ['stl10', 'stl10_64']:
            base_dataset = torchvision.datasets.STL10(root=data_dir, split='train', download=(not os.path.exists(os.path.join(data_dir, 'stl10_binary'))))
            train_labels = np.array(base_dataset.labels)
        
    noise_npy = np.load(poisons_path)
    best_noise = torch.from_numpy(noise_npy)

    # If the dataset is STL-10, resize the poison patches to 96x96
    if dataset == 'stl10':
        best_noise = F.interpolate(best_noise, size=(96, 96), mode='bilinear', align_corners=False)
    elif dataset == 'stl10_64':
        best_noise = F.interpolate(best_noise, size=(64, 64), mode='bilinear', align_corners=False)
    
    train_target_list = np.where(train_labels == label)[0]
    
    if last_n:
        # print(f"Poisoning last {poison_amount} images")
        train_target_list = train_target_list[-poison_amount:]
    elif index_list is not None:
        # print(f"Poisoning {len(index_list)} images from index list")
        train_target_list = np.intersect1d(train_target_list, index_list)
    else:
        # print(f"Poisoning {poison_amount} images from random selection")
        train_target_list = np.random.choice(train_target_list, poison_amount, replace=False)

    
    if dataset == 'stl10_64':
        forward_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), 
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        forward_transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    inverse_transform = transforms.Compose([transforms.Normalize((-1, -1, -1), (2, 2, 2)), 
                                            transforms.ToPILImage()])
    
    poison_indices = np.random.choice(train_target_list, poison_amount, replace=False)
    poison_tuples = [(inverse_transform(torch.clamp(apply_noise_patch(best_noise, forward_transform(base_dataset[i][0]), mode='add'), -1, 1)), 
                      label) for i in poison_indices]
    
    return poison_tuples, poison_indices, noise_npy

def apply_noise_patch(noise,images,offset_x=0,offset_y=0,mode='change',padding=20,position='fixed'):
    '''
    noise: torch.Tensor(1, 3, pat_size, pat_size)
    images: torch.Tensor(N, 3, img_size, img_size)
    outputs: torch.Tensor(N, 3, img_size, img_size)
    '''
    length = images.shape[2] - noise.shape[2]
    if position == 'fixed':
        wl = offset_x
        ht = offset_y
    else:
        wl = np.random.randint(padding,length-padding)
        ht = np.random.randint(padding,length-padding)
    if images.dim() == 3:
        noise_now = noise.clone()[0,:,:,:]
        wr = length-wl
        hb = length-ht
        m = nn.ZeroPad2d((wl, wr, ht, hb))
        if(mode == 'change'):
            images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
            images += m(noise_now)
        else:
            images += m(noise_now)
    else:
        for i in range(images.shape[0]):
            noise_now = noise.clone()
            wr = length-wl
            hb = length-ht
            m = nn.ZeroPad2d((wl, wr, ht, hb))
            if(mode == 'change'):
                images[i:i+1,:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
                images[i:i+1] += m(noise_now)
            else:
                images[i:i+1] += m(noise_now)
    return images

def get_poisoned_subset_GM(poisons_path):
    '''
    Function to load poisons from Gradient Matching attack
    '''
    with open(os.path.join(poisons_path, "poisons.pickle"), "rb") as handle:
        poison_tuples = pickle.load(handle)
        # logger.info(f"{len(poison_tuples)} poisons in this trial.")
        poisoned_label = poison_tuples[0][1]
    with open(os.path.join(poisons_path, "base_indices.pickle"), "rb") as handle:
        poison_indices = pickle.load(handle)

    return poison_tuples, poison_indices, poisoned_label

def process_args(args, rank):
    '''
    Function to process the arguments for distributed training
    '''
    # List of argument names to process
    arg_names = ['ebm_lang_steps', 'ebm_lang_temp', 'diff_train_steps', 'diff_purify_steps', 'diff_eta','ebm_name','diff_name','ebm_nf','diff_nf','num_images_narcissus']

    for arg_name in arg_names:
        arg_value = getattr(args, arg_name)

        # Check if the argument value is a list
        if isinstance(arg_value, list):
            try:
                # Try to get the value at the rank index
                setattr(args, arg_name, arg_value[rank])
            except IndexError:
                # If the index doesn't exist, set the argument value to None
                setattr(args, arg_name, None)

        # If any of the argument values are None, return None
        if getattr(args, arg_name) is None:
            return None

    return args