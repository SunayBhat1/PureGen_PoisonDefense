import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import os


#############
# Definitions
#############

dataset_dict = {'cifar10':{'image_dim': 32, 'num_classes': 10,'fid_batch_num': 60},
                'cinic10':{'image_dim': 32, 'num_classes': 10,'fid_batch_num': 60},
                'cincic10_imagenet':{'image_dim': 32, 'num_classes': 10,'fid_batch_num': 60},
                'tiny_imagenet':{'image_dim': 64, 'num_classes': 200,'fid_batch_num': 60},
                'stl10':{'image_dims': 96, 'num_classes': 10,'fid_batch_num': 10},
                'office_home':{'image_dim': 128, 'num_classes': 65,'fid_batch_num': 20},
                'caltech256': {'image_dim': 256, 'num_classes': 256, 'fid_batch_num': 5},
                'textures': {'image_dim': 256, 'num_classes': 47, 'fid_batch_num': 10},
                'flowers102': {'image_dim': 256, 'num_classes': 102, 'fid_batch_num': 10},
                'lfw_people': {'image_dim': 250, 'num_classes': 5749, 'fid_batch_num': 10},
                'fgvc_aircraft': {'image_dim': 256, 'num_classes': 100, 'fid_batch_num': 10},
                'food101': {'image_dim': 512, 'num_classes': 101, 'fid_batch_num': 10},
                'oxford_iiit_pet': {'image_dim': 256, 'num_classes': 37, 'fid_batch_num': 10},
                }

norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]

#############
# General Utils
#############

def get_train_data(dataset_type, data_dir, use_random_transform=False, img_resize=None, poisoned=False, poison_amount=500,poison_sz=32,poison_eps=8):

    img_dim = dataset_dict[dataset_type]['image_dim']

    ##############
    # Transforms #
    ##############

    transform = []

    if img_resize is not None:
        transform.append(transforms.Resize((img_resize, img_resize)))
        img_dim = img_resize

    if use_random_transform:
        transform.append(transforms.RandomCrop(img_dim, padding=img_dim//8))
        transform.append(transforms.RandomHorizontalFlip())

    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(norm_mean, norm_std))

    transform = transforms.Compose(transform)

    ##############
    # Load Data  #
    ##############

    # Not Poisoned
    if not poisoned:

        if dataset_type == 'cifar10':
            dataset = datasets.CIFAR10(data_dir, train=True, download=(not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py'))), transform=transform)

        elif dataset_type == 'cinic10':
            dataset = datasets.ImageFolder(os.path.join(data_dir, 'CINIC-10/train'), transform=transform)

        elif dataset_type == 'cincic10_imagenet':
            dataset = datasets.ImageFolder(os.path.join(data_dir, 'CINIC-10/train'), transform=transform)
            cifar_idxs = [idx for idx, (path, label) in enumerate(dataset.samples) if 'cifar10' not in path]

            dataset = Subset(dataset, cifar_idxs)

        elif dataset_type == 'tiny_imagenet':
            dataset = datasets.ImageFolder(os.path.join(data_dir, 'tiny-imagenet-200/train'), transform=transform)

        elif dataset_type == 'stl10':
            dataset = datasets.STL10(data_dir, split='unlabeled', download=(not os.path.exists(os.path.join(data_dir, 'stl10_binary'))), transform=transform)

        elif dataset_type == 'office_home':
            dataset = datasets.ImageFolder(os.path.join(data_dir, 'OfficeHomeDataset'), transform=transform)

            # Split the dataset into train and validation (first 90% is train, last 10% is validation)
            indices = list(range(len(dataset)))
            train_indices = indices[:int(0.9 * len(dataset))]
            dataset = Subset(dataset, train_indices)

        elif dataset_type == 'caltech256':
            dataset =datasets.ImageFolder((os.path.join(data_dir, '256_ObjectCategories')), transform=transform)

            # Split the dataset into train and validation (first 90% is train, last 10% is validation)
            indices = list(range(len(dataset)))
            train_indices = indices[:int(0.9 * len(dataset))]
            dataset = Subset(dataset, train_indices)

        elif dataset_type == 'textures':
            dataset = datasets.DTD(data_dir,split='train', download=(not os.path.exists(os.path.join(data_dir, 'dtd'))), transform=transform)
            
        elif dataset_type == 'flowers102':
            dataset = datasets.Flowers102(data_dir, split='train', download=(not os.path.exists(os.path.join(data_dir, 'flowers-102'))), transform=transform)

        elif dataset_type == 'lfw_people':
            dataset = datasets.LFWPeople(data_dir, split='train', download=(not os.path.exists(os.path.join(data_dir, 'lfw-people'))), transform=transform)
        elif dataset_type == 'fgvc_aircraft':
            dataset = datasets.FGVCAircraft(data_dir, split='train', download=(not os.path.exists(os.path.join(data_dir, 'fgvc-aircraft-2013b'))), transform=transform)
        elif dataset_type == 'food101':
            dataset = datasets.Food101(data_dir, split='train', download=(not os.path.exists(os.path.join(data_dir, 'food-101'))), transform=transform)
        elif dataset_type == 'oxford_iiit_pet':
            dataset = datasets.OxfordIIITPet(data_dir, split='trainval', download=(not os.path.exists(os.path.join(data_dir, 'oxford-iiit-pet'))), transform=transform)
        else:
            raise NotImplementedError
        
        return dataset
        
    else:

        if dataset_type == 'cifar10':
        
            poison_tuples_all, poison_indices_all = get_poisoned_subset_cifar10_narcissus(data_dir, poison_amount,'cifar10',poison_sz,poison_eps)
            cifar = True
            
        elif dataset_type == 'cinic10':
                
            poison_tuples_all, poison_indices_all = get_poisoned_subset_cifar10_narcissus(data_dir, poison_amount, cifar=False,poison_sz=poison_sz,poison_eps=poison_eps)
            cifar = False

        else:
            raise NotImplementedError

        dataset = Poisoned_Dataset(data_dir, transform=transform, num_per_label=9000,
                                    poison_tuple_list=poison_tuples_all, poison_indices=poison_indices_all,
                                    cifar=cifar)
    
        return dataset, poison_indices_all
    
def get_test_data(dataset_type, data_dir,img_resize=None):

    if img_resize is not None:
        transform = transforms.Compose([transforms.Resize((img_resize,img_resize)), transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

    if dataset_type == 'cifar10':
        dataset = datasets.CIFAR10(data_dir, train=False, download=(not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py'))), transform=transform)

    elif dataset_type == 'cinic10':
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'CINIC-10/test'), transform=transform)

    elif dataset_type == 'cincic10_imagenet':
            
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'CINIC-10/test'), transform=transform)
        cifar_idxs = [idx for idx, (path, label) in enumerate(dataset.samples) if 'cifar10' not in path]

        dataset = Subset(dataset, cifar_idxs)
    
    elif dataset_type == 'tiny_imagenet':
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'tiny-imagenet-200/val'), transform=transform)

    elif dataset_type == 'stl10':
        dataset = datasets.STL10(data_dir, split='test', download=(not os.path.exists(os.path.join(data_dir, 'stl10_binary'))), transform=transform)
        
    elif dataset_type == 'office_home':
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'OfficeHomeDataset'), transform=transform)

        # Split the dataset into train and validation (first 90% is train, last 10% is validation)
        indices = list(range(len(dataset)))
        val_indices = indices[int(0.9 * len(dataset)):]
        dataset = Subset(dataset, val_indices)

    elif dataset_type == 'caltech256':
        dataset =datasets.ImageFolder((os.path.join(data_dir, '256_ObjectCategories')), transform=transform)

        # Split the dataset into train and validation (first 90% is train, last 10% is validation)
        indices = list(range(len(dataset)))
        val_indices = indices[int(0.9 * len(dataset)):]
        dataset = Subset(dataset, val_indices)

    elif dataset_type == 'textures':
        dataset = datasets.DTD(data_dir,split='test', download=(not os.path.exists(os.path.join(data_dir, 'dtd'))), transform=transform)

    elif dataset_type == 'flowers102':
        dataset = datasets.Flowers102(data_dir, split='test', download=(not os.path.exists(os.path.join(data_dir, 'flowers-102'))), transform=transform)

    elif dataset_type == 'lfw_people':
        dataset = datasets.LFWPeople(data_dir, split='test', download=(not os.path.exists(os.path.join(data_dir, 'lfw-people'))), transform=transform)
    
    elif dataset_type == 'fgvc_aircraft':
        dataset = datasets.FGVCAircraft(data_dir, split='test', download=(not os.path.exists(os.path.join(data_dir, 'fgvc-aircraft-2013b'))), transform=transform)

    elif dataset_type == 'food101':
        dataset = datasets.Food101(data_dir, split='test', download=(not os.path.exists(os.path.join(data_dir, 'food-101'))), transform=transform)

    elif dataset_type == 'oxford_iiit_pet':
        dataset = datasets.OxfordIIITPet(data_dir, split='test', download=(not os.path.exists(os.path.join(data_dir, 'oxford-iiit-pet'))), transform=transform)

    else:
        raise NotImplementedError
    
    return dataset


#############
# EBM Poison Training Utils
#############

def get_poisoned_subset_cifar10_narcissus(data_dir, poison_amount,dataset='cifar10',size=32,eps=8,cifar=True):

    forward_transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    inverse_transform = transforms.Compose([transforms.Normalize((-1, -1, -1), (2, 2, 2)), 
                                            transforms.ToPILImage()])

    if dataset == 'cifar10':
        base_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False)
    elif dataset == 'cinic10':
        base_dataset = datasets.ImageFolder(os.path.join(data_dir, 'CINIC-10/train'))
        cifar_idxs = [idx for idx, (path, label) in enumerate(base_dataset.samples) if 'cifar10' in path]

    train_labels = np.array(base_dataset.targets)

    poison_tuples_all = []
    poison_indices_all = []

    for class_num in range(10):
        noise_npy = np.load(os.path.join(data_dir,f'Poisons/Narcissus/{dataset}/size={size}_eps={eps}/best_noise_lab{class_num}.npy'))
        best_noise = torch.from_numpy(noise_npy)

        if dataset == 'cifar10':
            train_target_list = np.where(train_labels == class_num)[0]
        elif dataset == 'cinic10':
            train_target_list = np.where(train_labels == class_num)[0]
            train_target_list = [idx for idx in train_target_list if idx in cifar_idxs]

        poison_indices = np.random.choice(train_target_list, poison_amount, replace=False)

        poison_tuples = [(inverse_transform(torch.clamp(forward_transform(base_dataset[i][0]) + best_noise[0], -1, 1)), class_num) for i in poison_indices]

        poison_tuples_all += poison_tuples
        poison_indices_all += list(poison_indices)
    
    return poison_tuples_all, poison_indices_all

class Poisoned_Dataset(Dataset):
    def __init__(self, path, transform=None, num_per_label=-1, 
                    poison_tuple_list=[], poison_indices=[], 
                    cifar=True
                 ):
               
        """
        Args:
        path: path to the dataset file.
        transform: transform to apply to the images.
        num_per_label: number of images per label to use in the dataset. If -1, uses all images.
        poison_tuple_list: list of tuples (image, label) to poison the dataset with.
        poison_indices: list of indices to poison the dataset with.
        transfer_subset: whether to use the transfer subset or the full dataset.
        """

        if cifar:
            dataset = datasets.CIFAR10(root=path, train=True, download=False)
        else:
            dataset = datasets.ImageFolder(os.path.join(path, 'CINIC-10/train'))

        self.img_label_list = [(img, label) for img, label in dataset]
        self.transform = transform
        # self.poison_mask = poison_mask
        self.class_labels = list(range(10))

        self.poison_indices = poison_indices
        self.poison_tuple_list = poison_tuple_list
        self.get_valid_indices(num_per_label, poison_indices)

    def get_valid_indices(self, num_per_label, poison_indices):
        num_per_label_dict = {label: 0 for label in self.class_labels}
        for pidx in poison_indices:
            img, label = self.img_label_list[pidx]
            if label in self.class_labels:
                num_per_label_dict[label] = num_per_label_dict.get(label, 0) + 1

        if num_per_label > 0:
            self.valid_indices = []
            for idx, (img, label) in enumerate(self.img_label_list):
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
            if self.transform is not None: img = self.transform(img)
            p = 1
            
        else:
            idx = self.valid_indices[index - len(self.poison_tuple_list)]
            img, label = self.img_label_list[idx]
            if self.transform is not None: img = self.transform(img)
        return img, label, index, p

class SubsetOfList(Dataset):
    def __init__(self, img_label_list, transform=None, start_idx=0, end_idx=1e10,
                    poison_tuple_list=[],
                    class_labels=list(range(10))):
        # logistics work for mixing data from CINIC10 with CIFAR10
        # suppose the poison samples are disjoint with the CINIC dataset
        self.img_label_list = img_label_list #torch.load(path)[subset]
        self.transform = transform
        self.poison_tuple_list = poison_tuple_list
        self.get_valid_list(start_idx, end_idx, class_labels)

    def get_valid_list(self, start_idx, end_idx, class_labels):
        # remove poisoned ones
        num_per_label_dict = {}
        selected_img_label_list = [] #[pt for pt in self.poison_tuple_list]
        if len(self.poison_tuple_list) > 0:
            poison_label = self.poison_tuple_list[0][1]
            # print("Poison label: {}".format(poison_label))
        else:
            poison_label = -1

        for idx, (img, label) in enumerate(self.img_label_list):
            if label not in class_labels:
                continue
            if label not in num_per_label_dict:
                num_per_label_dict[label] = 0
            if num_per_label_dict[label] >= start_idx and num_per_label_dict[label] < end_idx:
                if label == poison_label and num_per_label_dict[label] - start_idx < len(self.poison_tuple_list):
                    pass
                else:
                    selected_img_label_list.append([img, label])
            num_per_label_dict[label] += 1

        self.img_label_list = selected_img_label_list


    def __len__(self):
        return len(self.img_label_list) + len(self.poison_tuple_list)

    def __getitem__(self, index):
        if index < len(self.poison_tuple_list):
            img, label = self.poison_tuple_list[index]
        else:
            img, label = self.img_label_list[index-len(self.poison_tuple_list)]
            if self.transform is not None:
                img = self.transform(img)

        return img, label
