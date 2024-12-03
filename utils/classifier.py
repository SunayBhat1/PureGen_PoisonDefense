import torch
from torch import nn
import numpy as np
import time
import os
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import *
import torch.utils.data as data
from tqdm import tqdm
import pickle
from PIL import Image
import torch.nn.functional as F
import random

try: import torch_xla.core.xla_model as xm
except: pass

# Add parent directory to sys path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Classifiers import load_model

#############
# Variables #
#############

# Normalizations
# cifar_mean = (0.4914, 0.4822, 0.4465)
# cifar_std = (0.2023, 0.1994, 0.2010)
cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2471, 0.2435, 0.2616)

stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2471, 0.2435, 0.2616)

tinyimagenet_mean = (0.4802, 0.4481, 0.3975)
tinyimagenet_std = (0.2302, 0.2265, 0.2262)

# Dataset info dict
dataset_dict = {'cifar10':{'num_classes':10,'img_dim':32},
                'cinic10':{'num_classes':10,'img_dim':32},
                'tinyimagenet':{'num_classes':200,'img_dim':64},
                'stl10':{'num_classes':10,'img_dim':96},
                'stl10_64':{'num_classes':10,'img_dim':64},
                }

# dataset_info dict
dataset_info = {'from_scratch':{
                    'cifar10': {'num_classes': 10, 'num_per_label': 5000},
                    'cinic10': {'num_classes': 10, 'num_per_label': 9000},
                    'tinyimagenet': {'num_classes': 200, 'num_per_label': 500},
                    'stl10': {'num_classes': 10, 'num_per_label': 500},
                    'stl10_64': {'num_classes': 10, 'num_per_label': 500},
                    },
                'linear_transfer':{
                    'cifar10': {'num_classes': 10, 'num_per_label': 200},
                    },
                'fine_tune_transfer':{
                    'cifar10': {'num_classes': 10, 'num_per_label': 200},
                    },
                }

# Poison num_targets dict
poison_num_targets = {  'clean': {'Narcissus': 10},
                        'from_scratch': {'Narcissus': 10,
                                         'GradientMatching': 100,
                                         'NeuralTangent': 8,
                                        },
                        'fine_tune_transfer': {'Narcissus': 10,
                                        'BullseyePolytope': 50,
                                        },
                        'linear_transfer': {'BullseyePolytope': 50,
                                        'BullseyePolytope_Bench': 100,
                                        },
                        }

####################
#   General Utils  #
####################

def check_arg_errors(args):
    '''
    Check for errors in the arguments
    '''
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
    if args.ebm_filter is not None and args.data_key == 'Baseline':
        raise ValueError('EBM filter only supported when using a purified data key data')
    
def setup_directories(args):
    # Setup directories for remote server
    if args.remote_user is not None:
        args.data_dir = args.data_dir.replace('/home',f'/home/{args.remote_user}')
        args.output_dir = args.output_dir.replace('/home',f'/home/{args.remote_user}')

    # Create the output directory
    args.output_dir = os.path.join(args.output_dir,args.poison_mode.title(),args.poison_type.title())

    if not os.path.exists(args.output_dir): 
        os.makedirs(args.output_dir)

####################
# Train Data Utils #
####################

def get_train_data(args,target_index,device,ebm_model=None):

    test_trigger_loaders,poison_target_image, target_mask_label = None,None,None

    train_transforms = get_train_transforms(args)

    train_data, target_mask_label = get_base_poisoned_dataset(args,target_index,train_transforms,device,ebm_model)

    if 'HLB' in args.model and args.dataset in ['cifar10'] and args.baseline_defense == 'None':
        train_loader = train_data
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,num_workers=4)

    # Print training data details
    if args.poison_mode == 'clean' or args.poison_type == 'Neural_Tangent': p_count = 0
    else: p_count = sum(p.sum().item() for _, _, _, p in train_loader)

    if args.baseline_defense == 'Friendly':
        
        if args.poison_mode == 'from_scratch':
            if args.dataset in ['cifar10','cinic10']:
                train_transforms_no_augs = transforms.Compose([transforms.Normalize(mean=cifar_mean, std=cifar_std)])
            elif args.dataset in ['stl10','stl10_64']:
                train_transforms_no_augs = transforms.Compose([transforms.Normalize(mean=stl10_mean, std=stl10_std)])
            elif args.dataset == 'tinyimagenet':
                train_transforms_no_augs = transforms.Compose([transforms.Normalize(mean=tinyimagenet_mean, std=tinyimagenet_std)])
            else:
                raise ValueError('Friendly Defense not supported for this dataset')
            train_data_no_augs, _ = get_base_poisoned_dataset(args,target_index,train_transforms_no_augs,device)
        else:
            train_transforms_no_augs = transforms.Compose([transforms.Normalize(mean=cifar_mean, std=cifar_std)])
            train_data_no_augs, _ = get_base_poisoned_dataset(args,target_index,train_transforms_no_augs,device)

        train_loader_noaugs = torch.utils.data.DataLoader(train_data_no_augs, batch_size=args.batch_size, shuffle=True,num_workers=4)

        return train_data, train_loader, train_loader_noaugs, p_count, test_trigger_loaders,poison_target_image, target_mask_label
    
    return train_data, train_loader, p_count, test_trigger_loaders,poison_target_image, target_mask_label

def get_base_poisoned_dataset(args,target_index, train_transforms,device,ebm_model=None):

    # NTG Attack Data
    if args.poison_type == 'NeuralTangent':
        base_data = torch.load(os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset,'NTG',args.data_key + '.pt'))
        base_data = Simple_Dataset_Base(base_data, transforms=transforms.Compose([transforms.ToTensor()]))
        target_mask_label = None

    # Clean Data
    elif args.poison_mode == 'clean':
        if args.dataset == 'cifar10':
            base_data = CIFAR10(args.data_dir, train=True, download=(not os.path.exists(os.path.join(args.data_dir, 'cifar-10-batches-py'))), transform=transforms.ToTensor())
        elif args.dataset == 'cinic10':
            base_data = ImageFolder(os.path.join(args.data_dir, 'CINIC-10/train'), transform=transforms.ToTensor())
        elif args.dataset == 'tinyimagenet':
            base_data = ImageFolder(os.path.join(args.data_dir, 'tiny-imagenet-200/train'), transform=transforms.ToTensor())
        elif args.dataset == 'stl10':
            base_data = STL10(args.data_dir, split='train', download=(not os.path.exists(os.path.join(args.data_dir, 'stl10_binary'))), transform=transforms.ToTensor())
        else:
            raise Exception(f"Dataset {args.dataset} not supported for from_scratch or clean poison mode")

        _, _, target_mask_label = load_poisons(args,target_index)

    # Poison Data
    else:

        if args.ebm_filter is not None:
            unpurified_data = torch.load(os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset,'Baseline.pt'))

        if args.poison_mode == 'from_scratch' or args.poison_type == 'BullseyePolytope_Bench':
            base_data = torch.load(os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset,args.data_key + '.pt'))
        elif args.poison_mode in ['linear_transfer','fine_tune_transfer']:
            base_data = torch.load(os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset,'TransferBase',args.data_key + '.pt'))

        poison_tuple_list, poison_indices, target_mask_label = load_poisons(args,target_index)

        num_classes,num_per_class = dataset_info[args.poison_mode][args.dataset]['num_classes'],dataset_info[args.poison_mode][args.dataset]['num_per_label']

        if args.poison_mode == 'BullseyePolytope_Bench':
            base_data = PoisonedDataset_Bench(base_data, args.data_dir,poison_tuple_list,
                                                            2500,poison_indices)
        else:
            if args.poison_mode != 'from_scratch':
                num_per_class = args.num_per_class
            
            base_data = Poisoned_Dataset_Base(base_data,
                                                poison_tuple_list=poison_tuple_list,
                                                poison_indices = poison_indices,
                                                num_per_label=num_per_class, num_classes=num_classes,
                                                transforms=transforms.Compose([transforms.ToTensor()]),
                                                )

            if args.ebm_filter is not None:
                unpurified_data = Poisoned_Dataset_Base(unpurified_data,
                                                    poison_tuple_list=poison_tuple_list,
                                                    poison_indices = poison_indices,
                                                    num_per_label=num_per_class, num_classes=num_classes,
                                                    transforms=transforms.Compose([transforms.ToTensor()]),
                                                    )
            
                
    base_loader = data.DataLoader(base_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.poison_mode == 'clean' or args.poison_type == 'NeuralTangent':
        train_data = PoisonedDataset(base_loader, poisoned = False, transform=train_transforms)
    else:
        train_data = PoisonedDataset(base_loader, poisoned = True, transform=train_transforms)


    if args.ebm_filter is not None:
        unpurified_loader = torch.utils.data.DataLoader(unpurified_data, batch_size=256, shuffle=False,num_workers=4)
        unpurified_data = PoisonedDataset(unpurified_loader, poisoned = True, transform=train_transforms)
        train_data = replace_high_energy_samples(unpurified_data, train_data, ebm_model, args.ebm_filter,device)

    if 'HLB' in args.model and args.dataset == 'cifar10' and args.baseline_defense == 'None':
        aug = {'flip': args.hlb_flip}
        if args.hlb_translate is not None: aug['translate'] = args.hlb_translate
        if args.hlb_cutout is not None: aug['cutout'] = args.hlb_cutout
        train_data = CifarLoader(train_data.data, train=True, batch_size=args.batch_size, aug=aug, device=device,dataset_name=args.dataset)

    return train_data, target_mask_label

def replace_high_energy_samples(base_data, purified_data, ebm_model, purify_amount, device):

    forward_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    energy_list = []

    if xm.is_master_ordinal():
        pbar = tqdm(total=len(base_data.data), desc="Collecting Sample Energies")

    with torch.no_grad():
        for input, target, index, p in base_data.data:
            
            input = forward_norm(input).unsqueeze(0).to(device)       
            energy = ebm_model(input.to(device)).item()
            energy_list.append(energy)

            xm.mark_step()
            if xm.is_master_ordinal(): pbar.update(1)

    _, indices = torch.topk(torch.tensor(energy_list), int(purify_amount * len(energy_list)), largest=True)

    # Replace unpurified with purified for highest energiues
    xm.master_print(f"Replacing {len(indices)} high energy samples with purified samples")
    for i in indices:
        base_data.data[i] = purified_data.data[i]

    return base_data


### Data Loaders ###

class PoisonedDataset(data.Dataset):

    def __init__(self, base_loader, poisoned, transform=None):
        # Extract the features
        input_list, label_list, p_list, index_list = [], [], [], []

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        if not poisoned:
            start_idx = 0
        
        # Iterate through the base loader
        for batch in (base_loader):

            if poisoned:
                input, target, index, p = batch
            else:
                input, target = batch
                p = torch.zeros_like(target)
                index = torch.arange(start_idx, start_idx + input.size(0))

            input_list.extend([transforms.ToPILImage()(img.squeeze(0)) for img in list(torch.unbind(input, dim=0))])
            label_list.extend(list(torch.unbind(target, dim=0)))
            p_list.extend(list(torch.unbind(p, dim=0)))
            index_list.extend(list(torch.unbind(index, dim=0)))
                            
        self.data = [(img, label, index, p) for img, label, index, p in zip(input_list, label_list, index_list, p_list)]

        # Friendly Noise Perturbations
        self.set_perturbations()
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

        self.indices = list(range(len(self.data)))

    def set_perturbations(self, perturbations=None):
        if perturbations is None:
            self.perturbations = [None for x in range(len(self.data))]
        else:
            self.perturbations = perturbations
    
    def __getitem__(self, index):

        image, target, index, p = self.data[index]

        # Friendly Noise
        if self.perturbations[index] is not None:
            image = self.to_tensor(image)
            image = torch.clamp(image + self.perturbations[index], 0, 1)
            image = self.to_pil(image)
       
        if self.transform is not None:
            image = self.transform(image)

        return image, target, index, p

    def __len__(self):
        return len(self.data)

class Poisoned_Dataset_Base(data.Dataset):
    def __init__(self, base_dataset, poison_tuple_list, poison_indices, transforms, num_per_label=0, num_classes=10):
        """
        Args:
            path (str): The path to the file containing the image and label data
            num_per_label (int): The number of images per label to include in the dataset
            transfer (bool): Whether or not to load the dataset from a transfer learning file
        """

        self.base_dataset = base_dataset

        self.class_labels = list(range(num_classes))

        self.img_label_list = [(img, label) for img, label in base_dataset]

        self.transforms = transforms

        self.poison_indices = poison_indices
        self.poison_tuple_list = poison_tuple_list
        self.get_valid_indices(num_per_label, poison_indices)

    def get_valid_indices(self, num_per_label, poison_indices):
        """
        This method generates a list of valid indices for the dataset.  If num_per_label is 0, all indices are considered valid.

        Parameters:
            num_per_label (int): The maximum number of images per label to include in the valid_indices list.
            poison_indices (list): A list of indices of poisoned images.

        Modifies:
            self.valid_indices: A list of indices of valid (non-poisoned and within the num_per_label limit) images.
        """
        num_per_label_dict = {label: 0 for label in self.class_labels}
        for pidx in poison_indices:
            img, label = self.img_label_list[pidx]
            label = label.item()
            if label in self.class_labels:
                num_per_label_dict[label] = num_per_label_dict.get(label, 0) + 1

        if num_per_label > 0:
            self.valid_indices = []
            for idx, (img, label) in enumerate(self.img_label_list):
                label = label.item()
                if label in self.class_labels and idx not in poison_indices and num_per_label_dict.get(label, 0) < num_per_label:
                    self.valid_indices.append(idx)
                    num_per_label_dict[label] += 1
        else:
            self.valid_indices = list(range(len(self.img_label_list)))

    def __len__(self):
        return len(self.valid_indices) + len(self.poison_tuple_list)

    def __getitem__(self, index):
        p = 0
        if index < len(self.poison_tuple_list):
            img, label = self.poison_tuple_list[index]
            p = 1
        else:
            idx = self.valid_indices[index - len(self.poison_tuple_list)]
            img, label = self.img_label_list[idx]
        return self.transforms(img), label, index, p

class Simple_Dataset_Base(data.Dataset):
    def __init__(self, base_dataset, transforms):
        """
        Args:
            base_dataset (Dataset): The base dataset.
            transforms (callable): A function/transform that takes in an PIL image and returns a transformed version.
        """

        self.base_dataset = base_dataset
        self.img_label_list = [(img, label) for img, label in base_dataset]
        self.transforms = transforms
        self.valid_indices = list(range(len(self.img_label_list)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        idx = self.valid_indices[index]
        img, label = self.img_label_list[idx]
        return self.transforms(img), label

class Simple_Poison_Dataset(data.Dataset):
    def __init__(self, base_dataset, transforms):
        """
        Args:
            base_dataset (Dataset): The base dataset.
            transforms (callable): A function/transform that takes in an PIL image and returns a transformed version.
        """

        self.base_dataset = base_dataset
        self.img_label_list = [(img, label) for img, label in base_dataset]
        self.transforms = transforms
        self.valid_indices = list(range(len(self.img_label_list)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        idx = self.valid_indices[index]
        img, label = self.img_label_list[idx]
        return self.transforms(img), label

class PoisonedDataset_Bench(data.Dataset):
    def __init__(
        self, base_data, path, poison_instances, size=None, poison_indices=None,
    ):
        """poison instances should be a list of tuples of poison examples
        and their respective labels like
            [(x_0, y_0), (x_1, y_1) ...]
        """
        super(PoisonedDataset_Bench, self).__init__()
        self.trainset = base_data
        self.poison_instances = poison_instances
        self.poison_indices = np.array([]) if poison_indices is None else poison_indices
        self.dataset_size = size if size is not None else len(self.trainset)
        self.poisoned_label = (
            None if len(poison_instances) == 0 else poison_instances[0][1]
        )

        self.find_indices()
        
        self.to_tensor = transforms.ToTensor()
            
    def __getitem__(self, index):

        num_clean_samples = self.dataset_size - len(self.poison_instances)
            
        if index > num_clean_samples - 1:
            img, label = self.poison_instances[index - num_clean_samples]
            return self.to_tensor(img), label, index, 1 
        else:
            new_index = self.clean_indices[index]
            img, label = self.trainset[new_index]
            return self.to_tensor(img), label, index, 0

    def __len__(self):
        return self.dataset_size

    def find_indices(self):
        good_idx = np.array([])
        batch_tar = np.array(self.trainset.targets)
        num_classes = len(set(batch_tar))
        num_per_class = int(self.dataset_size / num_classes)
        for label in range(num_classes):
            all_idx_for_this_class = np.where(batch_tar == label)[0]
            all_idx_for_this_class = np.setdiff1d(
                all_idx_for_this_class, self.poison_indices
            )
            this_class_idx = all_idx_for_this_class[:num_per_class]
            if label == self.poisoned_label and len(self.poison_instances) > 0:
                num_clean = num_per_class - len(self.poison_instances)
                this_class_idx = this_class_idx[:num_clean]
            good_idx = np.concatenate((good_idx, this_class_idx))

        self.clean_indices = good_idx.astype(int)

class CifarLoader:

    def __init__(self, dataset, train=True, batch_size=500, aug=None, drop_last=None, shuffle=None, device=0,dataset_name='cifar10'):

        if train:
            self.images, self.labels, self.indices, self.p_values = zip(*dataset)

            self.images = [np.array(img).astype(np.uint8) for img in self.images]
            self.images = torch.tensor(np.array(self.images))
            self.labels = torch.tensor(self.labels)
            self.indices = torch.tensor(self.indices)
            self.p_values = torch.tensor(self.p_values)
        else:
            self.images = torch.tensor(dataset.data)
            if dataset_name in ['cifar10','cinic10']:
                self.labels = torch.tensor(dataset.targets)
            elif dataset_name == 'stl10':
                self.labels = torch.tensor(dataset.labels)
            else:
                raise Exception(f"Dataset {dataset_name} not supported for HLB Loader")


        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        if dataset_name == 'cifar10':
            self.normalize = transforms.Normalize(cifar_mean, cifar_std)
            self.denormalize = transforms.Normalize(
                                    tuple(-mean / std for mean, std in zip(cifar_mean, cifar_std)), 
                                    tuple(1 / std for std in cifar_std)
                                )
        elif dataset_name == 'cinic10':
            self.normalize = transforms.Normalize(cifar_mean, cifar_std)
            self.denormalize = transforms.Normalize(
                                    tuple(-mean / std for mean, std in zip(cifar_mean, cifar_std)), 
                                    tuple(1 / std for std in cifar_std)
                                )
        elif dataset_name == 'stl10':
            self.normalize = transforms.Normalize(stl10_mean, stl10_std)
            self.denormalize = transforms.Normalize(
                                    tuple(-mean / std for mean, std in zip(stl10_mean, stl10_std)), 
                                    tuple(1 / std for std in stl10_std)
                                )
        
        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate', 'cutout'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle
        self.device = device
        self.train = train

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        obj = {'images': self.images, 'labels': self.labels}
        torch.save(obj, path)

    def load(self, path):
        obj = torch.load(path)
        self.images = obj['images'].to(self.device)
        self.labels = obj['labels'].to(self.device)
        return self

    def augment(self, images):
        if self.aug.get('flip', False):
            images = batch_flip_lr(images)
        if self.aug.get('cutout', 0) > 0:
            images = batch_cutout(images, self.aug['cutout'])
        if self.aug.get('translate', 0) > 0:
            # Apply translation in minibatches in order to save memory
            images = torch.cat([batch_translate(image_batch, self.aug['translate'])
                                for image_batch in images.split(5000)])
        return images

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else int(np.ceil(len(self.images)/self.batch_size))

    def __iter__(self):
        images = self.augment(self.normalize(self.images))
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            if self.train:
                yield (images[idxs], self.labels[idxs], self.indices[idxs], self.p_values[idxs])
            else:
                yield (images[idxs], self.labels[idxs])


###################
# Test Data Utils #
###################
                
def get_test_dataset(args):

    transform = get_test_transforms(args)

    if args.dataset == 'cifar10':
        test_data = CIFAR10(args.data_dir, train=False, download=(not os.path.exists(os.path.join(args.data_dir, 'cifar-10-batches-py'))), transform=transform)
    elif args.dataset == 'cinic10':
        test_data = ImageFolder(os.path.join(args.data_dir, 'CINIC-10/test'), transform=transform)
    elif args.dataset == 'tinyimagenet':
        test_data = TinyImageNetValDataset(os.path.join(args.data_dir, 'tiny-imagenet-200'), transform=transform)
    elif args.dataset == 'stl10':
        test_data = STL10(args.data_dir, split='test', download=(not os.path.exists(os.path.join(args.data_dir, 'stl10_binary'))), transform=transform)
    else:
        raise Exception(f"Dataset {args.dataset} not supported in function get_test_dataset")

    if 'HLB' in args.model and args.dataset in ['cifar10'] and args.baseline_defense == 'None':
        test_loader = CifarLoader(test_data, train=False, batch_size=1000,dataset_name=args.dataset)
    else:
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=128,num_workers=4)

    return test_loader, transform

def get_test_transforms(args):
    if args.dataset in ['cifar10','cinic10']:
        test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cifar_mean, std=cifar_std)])
    elif args.dataset == 'stl10':
        test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=stl10_mean, std=stl10_std)])
    elif args.dataset == 'stl10_64':
        test_transforms = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(), transforms.Normalize(mean=stl10_mean, std=stl10_std)])
    elif args.dataset == 'tinyimagenet':
        test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=tinyimagenet_mean, std=tinyimagenet_std)])

    return test_transforms


class TinyImageNetValDataset(data.Dataset):
    def __init__(self, tiny_imagenet_folder, transform=None):
        """
        Initializes the dataset loader for the Tiny ImageNet validation set.
        
        Args:
            tiny_imagenet_folder (string): Directory with all the Tiny ImageNet dataset, including 'val' folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.tiny_imagenet_folder = tiny_imagenet_folder
        self.transform = transform

        # Load class indices
        self.classes = sorted(item for item in os.listdir(os.path.join(tiny_imagenet_folder, 'train')) if item != '.DS_Store')
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Load image paths and labels
        self.image_paths, self.labels = self._load_labels_from_file(os.path.join(tiny_imagenet_folder, 'val', 'val_annotations.txt'))

    def _load_labels_from_file(self, labels_file):
        """
        Loads image paths and labels from the val_annotations.txt file.
        
        Args: labels_file (string): Path to the val_annotations.txt file.
        Returns: tuple: A tuple containing lists of image paths and their corresponding labels.
        """
        label_data = []
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                # Append the image path and class name
                label_data.append((parts[0], parts[1]))

        # Replace class names with indices and form full paths
        labels = [self.class_to_idx[class_name] for _, class_name in label_data]
        image_paths = [os.path.join(self.tiny_imagenet_folder, 'val', 'images', path) for path, _ in label_data]
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

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

# Data Loader for CIFAR-10 with single poison trigger
class Trigger_Test_Dataset(data.Dataset):
    def __init__(self, dataset,indices,noise,target,transform,clean=False):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.target = target
        self.transform = transform
        self.clean = clean

    def __getitem__(self, idx):
        forward_transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        inverse_transform = transforms.Compose([transforms.Normalize((-1, -1, -1), (2, 2, 2)), 
                                                transforms.ToPILImage()])
        image, label = self.dataset[self.indices[idx]]
        if not self.clean:
            image = forward_transform(image)
            image = torch.clamp(apply_noise_patch(self.noise, image, mode='add'), -1, 1)
            image = inverse_transform(image)
        if self.transform is not None:
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.indices)
    
### HLB Loader ###
    
def make_random_square_masks(inputs, size):
    is_even = int(size % 2 == 0)
    n,c,h,w = inputs.shape

    # seed top-left corners of squares to cutout boxes from, in one dimension each
    corner_y = torch.randint(0, h-size+1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w-size+1, size=(n,), device=inputs.device)

    # measure distance, using the center as a reference point
    corner_y_dists = torch.arange(h, device=inputs.device).view(1, 1, h, 1) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(1, 1, 1, w) - corner_x.view(-1, 1, 1, 1)
    
    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)

    final_mask = mask_y * mask_x

    return final_mask

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(inputs, crop_size):
    crop_mask = make_random_square_masks(inputs, crop_size)
    cropped_batch = torch.masked_select(inputs, crop_mask)
    return cropped_batch.view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)

def batch_translate(inputs, translate):
    width = inputs.shape[-2]
    padded_inputs = F.pad(inputs, (translate,)*4, 'reflect', value=0)
    return batch_crop(padded_inputs, width)

def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)


###############
# Transform Utils
#############
    
def get_train_transforms(args):
    if args.poison_type == 'NeuralTangent':

        train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((cifar_mean), (cifar_std)),
            ])
        
        return train_transforms
    elif args.poison_mode == 'from_scratch' or args.poison_type in ['BullseyePolytope_Bench','Narcissus']:
        if args.dataset in ['cifar10','cinic10']:
            train_transforms = [transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()]
        elif args.dataset == 'stl10':
            train_transforms = [transforms.RandomCrop(96, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()]
        elif args.dataset == 'tinyimagenet':
            train_transforms = [transforms.RandomCrop(64, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()]
        else:
            raise Exception(f"Dataset {args.dataset} transforms not supported for from_scratch poison mode")
    elif args.poison_mode in ['linear_transfer','fine_tune_transfer'] and args.poison_type == 'BullseyePolytope':
        train_transforms = [transforms.ToTensor()]

    if args.baseline_defense == 'Friendly':
        if "uniform" in args.friendly_noise_type:
            train_transforms.append(UniformNoise(eps=args.friendly_noise_eps / 255))
        if "gaussian" in args.friendly_noise_type:
            train_transforms.append(GaussianNoise(eps=args.friendly_noise_eps / 255))
        if "bernoulli" in args.friendly_noise_type:
            train_transforms.append(BernoulliNoise(eps=args.friendly_noise_eps / 255))


    if args.dataset in ['cifar10','cinic10']:
        train_transforms.append(transforms.Normalize(cifar_mean, cifar_std))
    elif args.dataset in ['stl10','stl10_64']:
        train_transforms.append(transforms.Normalize(stl10_mean, stl10_std))
    elif args.dataset == 'tinyimagenet':
        train_transforms.append(transforms.Normalize(tinyimagenet_mean, tinyimagenet_std))
    else:
        raise Exception(f"Dataset {args.dataset} transforms not supported for from_scratch poison mode")

    if args.baseline_defense == 'Friendly' or args.aug_rand_transforms:
        img_size = dataset_dict[args.dataset]['img_dim']
        train_transforms.append(RandomTransform(**dict(source_size=img_size, target_size=img_size, shift=8, fliplr=True), mode='bilinear'))

    if args.aug_cutout:
        train_transforms.append(Cutout(n_holes=8, length=8))

    return transforms.Compose(train_transforms)

class UniformNoise(object):
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, tensor):
        out = tensor + torch.rand(tensor.size()) * self.eps * 2 -self.eps
        return out

class GaussianNoise(object):
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, tensor):
        out = tensor + torch.randn(tensor.size()) * self.eps
        return out

class BernoulliNoise(object):
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, tensor):
        noise = (torch.rand(tensor.size()) > 0.5).float() * 2 - 1
        out = tensor + noise * self.eps
        return out

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
class RandomTransform(torch.nn.Module):
    """Crop the given batch of tensors at a random location.

    As discussed in https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5
    """

    def __init__(self, source_size, target_size, shift=8, fliplr=True, flipud=False, mode='bilinear', align=True):
        """Args: source and target size."""
        super().__init__()
        self.grid = self.build_grid(source_size, target_size)
        self.delta = torch.linspace(0, 1, source_size)[shift]
        self.fliplr = fliplr
        self.flipud = flipud

        self.mode = mode
        self.align = True

    @staticmethod
    def build_grid(source_size, target_size):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        k = float(target_size) / float(source_size)
        direct = torch.linspace(-1, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
        return full

    def random_crop_grid(self, x, randgen=None):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        grid = self.grid.repeat(x.size(0), 1, 1, 1).clone().detach()
        grid = grid.to(device=x.device, dtype=x.dtype)
        if randgen is None:
            randgen = torch.rand(x.shape[0], 4, device=x.device, dtype=x.dtype)

        # Add random shifts by x
        x_shift = (randgen[:, 0] - 0.5) * 2 * self.delta
        grid[:, :, :, 0] = grid[:, :, :, 0] + x_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))
        # Add random shifts by y
        y_shift = (randgen[:, 1] - 0.5) * 2 * self.delta
        grid[:, :, :, 1] = grid[:, :, :, 1] + y_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))

        if self.fliplr:
            grid[randgen[:, 2] > 0.5, :, :, 0] *= -1
        if self.flipud:
            grid[randgen[:, 3] > 0.5, :, :, 1] *= -1
        return grid

    def forward(self, x, randgen=None):
        x = torch.unsqueeze(x, 0)
        # Make a random shift grid for each batch
        grid_shifted = self.random_crop_grid(x, randgen)
        # Sample using grid sample
        return torch.squeeze(F.grid_sample(x, grid_shifted, align_corners=self.align, mode=self.mode))


#############
# Model Utils
#############


def load_target_network(args,device):

    num_classes = dataset_dict[args.dataset]['num_classes']
    img_dim = dataset_dict[args.dataset]['img_dim']

    if args.poison_mode in ['from_scratch','clean']:
        target_net = load_model(args.model, num_classes=num_classes, img_size=img_dim)
    elif args.poison_type == 'BullseyePolytope':
        target_net = load_model(args.model, eval_bn=True)
    elif args.poison_type == 'BullseyePolytope_Bench':
        target_net = load_model(args.model, num_classes=100, eval_bn=True)
    else:
        target_net = load_model(args.model)

    # Load state dict for transfer learning
    if args.poison_mode in ['linear_transfer','fine_tune_transfer']:
            
        # Use benchmark if running BullseyePolytope_Bench
        if args.poison_type == 'BullseyePolytope_Bench':
            model_path = 'ResNet18_CIFAR100.pth'
        else:
            model_path = args.model_path

        state_dict_path = os.path.join(args.data_dir, 'PureGen_Models','transfer_models', model_path)
        state_dict_module = torch.load(state_dict_path,map_location=torch.device('cpu'))['net']
        state_dict = {}
        for k,v in state_dict_module.items():
            state_dict[k.replace('module.', '')] = v
        target_net.load_state_dict(state_dict)

        if args.verbose: print(f'Loaded the target network from {state_dict_path}')

    # Move target_net to device
    if 'HLB' in args.model:
        target_net = target_net.to(device).to(memory_format=torch.channels_last)
    else:
        target_net = target_net.to(device)

    # Reinit Linear layer for transfer learning
    if args.poison_mode in ['linear_transfer','fine_tune_transfer'] and args.reinit_linear:
        if args.model == 'ResNet18':
            target_net.linear = nn.Linear(512, 10).to(device)
        elif args.model == 'DenseNet121':
            target_net.linear = nn.Linear(1024, 10).to(device)
        elif args.model == 'MobileNetV2':
            target_net.linear = nn.Linear(1280, 10).to(device)
        if args.verbose: print(f'Reinitialized the linear layer of the target network')

    return target_net


###################
# Optimizer Utils #
###################

def get_optimizer(args,target_net):

    if args.poison_mode in ['from_scratch','clean']:
        
        ### HLB Optimizers ###
        if args.model in ['HLB_S','HLB_M','HLB_L']:
            kilostep_scale = 1024 * (1 + 1 / (1 - args.momentum))
            lr = args.lr / kilostep_scale # un-decoupled learning rate for PyTorch SGD
            wd = args.weight_decay * args.batch_size / kilostep_scale
            lr_biases = lr * args.bias_scaler

            norm_biases = [p for k, p in target_net.named_parameters() if 'norm' in k and p.requires_grad]
            other_params = [p for k, p in target_net.named_parameters() if 'norm' not in k and p.requires_grad]
            param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                            dict(params=other_params, lr=lr, weight_decay=wd/lr)]
            optimizer = torch.optim.SGD(param_configs, momentum=args.momentum, nesterov=True)    

        ### Standard Optimizers ###
        elif args.optim == 'sgd':
            if args.model == 'ResNet18_HLB':
                optimizer = torch.optim.SGD(target_net.parameters(), lr=args.lr/args.batch_size, momentum=args.momentum, nesterov=True,
                                    weight_decay=args.weight_decay*args.batch_size)
            else:
                optimizer = torch.optim.SGD(target_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        elif args.optim == 'adam':
            optimizer = torch.optim.Adam(target_net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        
        elif args.optim == 'adamw':
            optimizer = torch.optim.AdamW(target_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        elif args.optim == 'sgd_gm':
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in target_net.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in target_net.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = torch.optim.SGD(grouped_parameters, lr=args.lr, momentum=args.momentum, nesterov=True)

        else:
            raise Exception(f"Optimizer {args.optim} not supported")

    elif args.poison_mode in ['linear_transfer','fine_tune_transfer']:

        if args.poison_mode == 'fine_tune_transfer':
            params = target_net.parameters()
        else:
            params = target_net.get_penultimate_params_list()

        if args.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    return optimizer

def get_scheduler(args,optimizer,len_data):
    if 'HLB' in args.model:
        if args.dataset == 'cifar10' and args.baseline_defense == 'None':
            total_train_steps = np.ceil(len_data * args.epochs)
        else:
            total_train_steps = np.ceil(len_data / args.batch_size * args.epochs) + args.epochs
        lr_schedule = np.interp(np.arange(1+total_train_steps),
                            [0, int(0.2 * total_train_steps), total_train_steps],
                            [0.2, 1, 0]) # triangular learning rate schedule
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
        if args.verbose: print(f'Loaded the HLB scheduler with {total_train_steps} steps')

    else:
        if args.sched == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay, gamma=0.1)
        elif args.sched == 'cosine':
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len_data, eta_min=1e-5)
            scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler=base_scheduler)

    return scheduler

#############
# Poison Utils
#############

def load_poisons(args,target_index):
    """
    This function loads the poison data from a file.
    Parameters:
        args (argparse.Namespace): The command-line arguments.
    Returns:
        tuple: A tuple containing the poison data, the indices of the poison data, and the target of the poison attack.
    """

    subfolder = os.path.join(args.data_dir,'PureGen_PurifiedData',args.dataset,'Poisons',args.poison_mode)

    if args.poison_type == 'GradientMatching':
        load_dir = os.path.join(args.data_dir, subfolder, 'GradientMatching')
    elif args.poison_type == 'Narcissus':
        load_dir = os.path.join(args.data_dir, subfolder, f'Narcissus/size={args.noise_sz_narcissus}_eps={args.noise_eps_narcissus}_num={args.num_images_narcissus}')
    elif args.poison_type == 'BullseyePolytope':
        load_dir = os.path.join(args.data_dir, subfolder, f'BullseyePolytope/imgs={args.num_images_bp}_iters={args.iters_bp}_repeat={args.net_repeat_bp}')
    elif args.poison_type == 'BullseyePolytope_Bench':
        load_dir = os.path.join(args.data_dir, subfolder, f'BullseyePolytopeBench/imgs={args.num_images_bp}')

    # Load the poison_tuple_list, poison_indices, and target
    poison_tuple_list, poison_indices, target = torch.load(os.path.join(load_dir, args.data_key, f'{target_index}.pth'))

    return poison_tuple_list, poison_indices, target
    

def get_poisons_target(args,target_index,test_transforms,target_mask=None):
    if args.poison_type =='GradientMatching':
        if args.dataset == 'cifar10':
            target_img, target_orig_label = get_target_poison(os.path.join(args.data_dir,f'Poisons/GradientMatching/{args.dataset}/{target_index}'), test_transforms)
        elif args.dataset == 'tinyimagenet':
            target_img, target_orig_label = get_target_poison(os.path.join(args.data_dir,f'Poisons/GradientMatching/{args.dataset}/ResNet34_250/{target_index}'), test_transforms)
        else:
            raise Exception(f"Dataset {args.dataset} not supported for GradientMatching poison mode")
        
        return target_img, target_orig_label
    
    elif args.poison_type == 'BullseyePolytope':
        target_img = fetch_target_polytope(6, target_index, args.num_per_class, path=os.path.join(args.data_dir,'CIFAR10_TRAIN_Split.pth'), subset='others',transform=test_transforms)
        return target_img, 6
    
    elif args.poison_type == 'BullseyePolytope_Bench':
        target_img, target_orig_label = get_target_poison(os.path.join(args.data_dir,f'Poisons/Transfer_Bench/bp_poisons/num_poisons={args.num_images_bp}/{target_index}'), test_transforms)
        return target_img, target_orig_label

    elif args.poison_type == 'Narcissus':
        # Target Test Sets
        test_trigger_loaders = {}
        for i in range(1,3):
            test_data_trigger = get_target_narcissus(args.data_dir, args.dataset, target_mask, target_index, test_transforms, multi_test=i)
            test_trigger_loaders[i] = torch.utils.data.DataLoader(test_data_trigger, batch_size=128)

        return test_trigger_loaders

def get_target_narcissus(data_dir, dataset, poison, label, transform_test, multi_test=1, clean=False): 
    # noise_npy = np.load(poisons_path)
    best_noise = torch.from_numpy(poison)
    if dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=(not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py'))))
        test_labels = np.array(test_dataset.targets)
    elif dataset == 'cinic10':
        test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'CINIC-10/test'))
        test_labels = np.array(test_dataset.targets)
    elif dataset == 'tiny-imagenet':
        test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'tiny-imagenet-200/test'))
        test_labels = np.array(test_dataset.targets)
    elif dataset in ['stl10','stl10_64']:
        test_dataset = torchvision.datasets.STL10(root=data_dir, split='test', download=(not os.path.exists(os.path.join(data_dir, 'stl10_binary'))))
        test_labels = np.array(test_dataset.labels)
    else:
        raise Exception(f"Dataset {dataset} not supported for Narcissus poison mode Test Data")
    
    test_target_list = np.where(test_labels != label)[0]
    return Trigger_Test_Dataset(test_dataset, test_target_list, best_noise * multi_test, label, transform=transform_test, clean=clean)

def fetch_target_polytope(target_label, target_index, start_idx, path, subset, transform=None):
    """
    Fetch the "target_index"-th target, counting starts from start_idx

    Args:
        target_label (int): The label of the target to fetch
        target_index (int): The index of the target to fetch, counting starts from start_idx
        start_idx (int): The starting index to count from
        path (str): The path to the file containing the image and label data
        subset (str): The subset of the data to fetch the target from

    Returns:
        numpy.ndarray or torch.Tensor: The image data for the fetched target, with an added batch dimension

    Raises:
        Exception: If the target with the given index exceeds the number of total samples in the subset
    """

    img_label_list = torch.load(path)[subset]
    counter = 0
    for idx, (img, label) in enumerate(img_label_list):
        if label == target_label:
            counter += 1
            if counter == (target_index + start_idx + 1):
                if transform is not None:
                    return transform(img)[None, :, :, :]
                else:
                    return np.array(img)[None, :, :, :]
    raise Exception("Target with index {} exceeds number of total samples (should be less than {})".format(
        target_index, len(img_label_list) / 10 - start_idx))


def get_target_poison(poisons_path, transform_test):
    # get the target image from pickled file

    with open(os.path.join(poisons_path, "target.pickle"), "rb") as handle:
        target_img_pil,target_class = pickle.load(handle)
    target_img = transform_test(target_img_pil)
    return target_img, target_class

