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

import torch.nn.functional as F
import random

try: import torch_xla.core.xla_model as xm
except: pass

# Normalization
cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2023, 0.1994, 0.2010)

cifar_mean_gm = (0.4914, 0.4822, 0.4465)
cifar_std_gm = (0.2471, 0.2435, 0.2616)

stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2471, 0.2435, 0.2616)

# dataset_info dict
dataset_info = {'from_scratch':{
                    'cifar10': {'num_classes': 10, 'num_per_label': 5000},
                    'cinic10': {'num_classes': 10, 'num_per_label': 9000},
                    'tiny-imagenet': {'num_classes': 200, 'num_per_label': 500},
                    'stl10': {'num_classes': 10, 'num_per_label': 500},
                    },
                'transfer':{
                    'cifar10': {'num_classes': 10, 'num_per_label': 200},
                    },
                }
        

####################
# Train Data Utils #
####################

def get_base_poisoned_dataset(args,target_index, train_transforms,device):

    # Load the train data
    if args.no_poison:
        if args.poison_mode == 'from_scratch':
            if args.dataset == 'cifar10':
                base_data = CIFAR10(args.data_dir, train=True, download=(not os.path.exists(os.path.join(args.data_dir, 'cifar-10-batches-py'))), transform=transforms.ToTensor())
            elif args.dataset == 'cinic10':
                base_data = ImageFolder(os.path.join(args.data_dir, 'CINIC-10/train'), transform=transforms.ToTensor())
            elif args.dataset == 'tiny-imagenet':
                base_data = ImageFolder(os.path.join(args.data_dir, 'tiny-imagenet-200/train'), transform=transforms.ToTensor())
            elif args.dataset == 'stl10':
                base_data = STL10(args.data_dir, split='train', download=(not os.path.exists(os.path.join(args.data_dir, 'stl10_binary'))), transform=transforms.ToTensor())
            else:
                raise Exception(f"Dataset {args.dataset} not supported for from_scratch poison mode")
        elif args.poison_mode == 'transfer':
            if args.dataset == 'cifar10':
                base_data = torch.load(os.path.join(args.data_dir,'CIFAR10_TRAIN_Split.pth'))['others']
            else:
                raise Exception(f"Dataset {args.dataset} not supported for transfer poison mode")
            
        target_mask_label = None

    else:

        base_data = torch.load(os.path.join(args.data_dir,'PureDefense',args.dataset,args.data_key + '.pt'))

        poison_tuple_list, poison_indices, target_mask_label = load_poisons(args,target_index)

        num_classes,num_per_label = dataset_info[args.poison_mode][args.dataset]['num_classes'],dataset_info[args.poison_mode][args.dataset]['num_per_label']

        if args.poison_mode == 'from_scratch':

            base_data = Poisoned_Dataset_Base(base_data,
                                                poison_tuple_list=poison_tuple_list,
                                                poison_indices = poison_indices,
                                                num_per_label=num_per_label, num_classes=num_classes,
                                                transforms=transforms.Compose([transforms.ToTensor()]),
                                                )
        elif args.poison_mode == 'transfer':
            # if poison_type == 'BullseyePolytope':
            #     base_data = Poisoned_Dataset_Base(os.path.join(args.data_dir,'CIFAR10_TRAIN_Split.pth'), 
            #                                             dataset=args.dataset,
            #                                             num_per_label=args.num_per_class_bp,
            #                                             poison_tuple_list= poison_tuple_list,
            #                                             poison_indices = poison_indices,
            #                                             transfer=True,
            #                                             )
            # elif poison_type == 'BullseyePolytope_Bench':
            #     base_data = PoisonedDataset_Bench(args.data_dir,args.dataset,poison_tuple_list,
            #                                                 2500,poison_indices)
                
            if args.poison_type == 'Narcissus':
                base_data = Poisoned_Dataset_Base(base_data, 
                                                    poison_tuple_list= poison_tuple_list,
                                                    poison_indices = poison_indices,
                                                    num_per_label=args.num_per_class_narcissus,
                                                    transfer=True,
                                                )
                
    base_loader = data.DataLoader(base_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_data = PoisonedDataset(base_loader, poisoned = not args.no_poison, transform=train_transforms)

    if 'HLB' in args.model and args.dataset == 'cifar10':
        aug = {'flip': args.hlb_flip}
        if args.hlb_translate is not None: aug['translate'] = args.hlb_translate
        if args.hlb_cutout is not None: aug['cutout'] = args.hlb_cutout
        train_data = CifarLoader(train_data.data, train=True, batch_size=args.batch_size, aug=aug, device=device,dataset_name=args.dataset)

    return train_data, target_mask_label

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

class PoisonedDataset_Bench(data.Dataset):
    def __init__(
        self, path, poison_instances, size=None, poison_indices=None,
    ):
        """poison instances should be a list of tuples of poison examples
        and their respective labels like
            [(x_0, y_0), (x_1, y_1) ...]
        """
        super(PoisonedDataset_Bench, self).__init__()
        self.trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=(not os.path.exists(os.path.join(path, 'cifar-10-batches-py'))))
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
                
def get_test_dataset(args, transform):
    if args.dataset == 'cifar10':
        test_data = CIFAR10(args.data_dir, train=False, download=(not os.path.exists(os.path.join(args.data_dir, 'cifar-10-batches-py'))), transform=transform)
    elif args.dataset == 'cinic10':
        test_data = ImageFolder(os.path.join(args.data_dir, 'CINIC-10/test'), transform=transform)
    elif args.dataset == 'tiny-imagenet':
        test_data = ImageFolder(os.path.join(args.data_dir, 'tiny-imagenet-200/test'), transform=transform)
    elif args.dataset == 'stl10':
        test_data = STL10(args.data_dir, split='test', download=(not os.path.exists(os.path.join(args.data_dir, 'stl10_binary'))), transform=transform)

    return test_data

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
    if args.poison_mode == 'from_scratch' or args.poison_type in ['BullseyePolytope_Bench','Narcissus']:
        if args.dataset in ['cifar10','cinic10']:
            train_transforms = [transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()]
        elif args.dataset == 'stl10':
            train_transforms = [transforms.RandomCrop(96, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()]
        elif args.dataset == 'tiny-imagenet':
            train_transforms = [transforms.RandomCrop(64, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()]
        else:
            raise Exception(f"Dataset {args.dataset} transforms not supported for from_scratch poison mode")
    elif args.poison_mode == 'transfer' and args.poison_type == 'BullseyePolytope':
        train_transforms = [transforms.ToTensor()]

    if args.add_rand_noise and args.defense != 'Friendly':
        if args.verbose: print(f'Adding {args.rand_noise_type} noise with eps {args.rand_noise_eps}')
        if "uniform" in args.rand_noise_type:
            train_transforms.append(UniformNoise(eps=args.rand_noise_eps / 255))
        if "gaussian" in args.rand_noise_type:
            train_transforms.append(GaussianNoise(eps=args.rand_noise_eps / 255))
        if "bernoulli" in args.rand_noise_type:
            train_transforms.append(BernoulliNoise(eps=args.rand_noise_eps / 255))

    if args.baseline_defense == 'Friendly':
        if "uniform" in args.friendly_noise_type:
            train_transforms.append(UniformNoise(eps=args.friendly_noise_eps / 255))
        if "gaussian" in args.friendly_noise_type:
            train_transforms.append(GaussianNoise(eps=args.friendly_noise_eps / 255))
        if "bernoulli" in args.friendly_noise_type:
            train_transforms.append(BernoulliNoise(eps=args.friendly_noise_eps / 255))

    if args.poison_mode == 'from_scratch':
        if args.dataset == 'cifar10':
            train_transforms.append(transforms.Normalize(cifar_mean_gm, cifar_std_gm))
        elif args.dataset == 'cinic10':
            train_transforms.append(transforms.Normalize(cifar_mean, cifar_std))
        elif args.dataset == 'stl10':
            train_transforms.append(transforms.Normalize(stl10_mean, stl10_std))
    elif args.poison_mode == 'transfer':
        train_transforms.append(transforms.Normalize(cifar_mean, cifar_std))

    if args.baseline_defense == 'Friendly' or args.aug_rand_transforms:
        train_transforms.append(RandomTransform(**dict(source_size=32, target_size=32, shift=8, fliplr=True), mode='bilinear'))

    if args.aug_cutout:
        train_transforms.append(Cutout(n_holes=8, length=8))
        

    return transforms.Compose(train_transforms)


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

    subfolder = os.path.join(args.data_dir,'PureDefense',args.dataset,'Poisons')

    if args.poison_type == 'Gradient_Matching':
        load_dir = os.path.join(args.data_dir, subfolder, 'Gradient_Matching')
    elif args.poison_type == 'Narcissus':
        load_dir = os.path.join(args.data_dir, subfolder, f'Narcissus/size={args.noise_sz_narcissus}_eps={args.noise_eps_narcissus}')
    elif args.poison_type == 'BullseyePolytope':
        if args.fine_tune: bp_subpath = 'end2end-training'
        else: bp_subpath = 'linear-transfer-learning'
        if args.num_images_bp == 5: bp_subpath = os.path.join(bp_subpath, f'mean-{args.net_repeat_bp}Repeat')
        else: bp_subpath = os.path.join(bp_subpath, f'mean')
        load_dir = os.path.join(args.data_dir, subfolder, f'Bullseye_Polytope/{args.num_images_bp}-imgs/{bp_subpath}/{args.iters_bp}-iters')
    elif args.poison_type == 'BullseyePolytope_Bench':
        load_dir = os.path.join(args.data_dir, subfolder, f'Transfer_Bench/bp_poisons/{args.num_images_bp}-imgs')

    # Load the poison_tuple_list, poison_indices, and target
    poison_tuple_list, poison_indices, target = torch.load(os.path.join(load_dir, args.data_key, f'{target_index}.pth'))

    return poison_tuple_list, poison_indices, target
    

def get_poisons_target(args,target_index,test_transforms,target_mask=None):
    if args.poison_type =='Gradient_Matching':
        target_img, target_orig_label = get_target_poison(os.path.join(args.data_dir,f'Poisons/Gradient_Matching/{target_index}'), test_transforms)
        return target_img, target_orig_label
    
    elif args.poison_type == 'BullseyePolytope':
        target_img = fetch_target_polytope(6, target_index, args.num_per_class_bp, path=os.path.join(args.data_dir,'CIFAR10_TRAIN_Split.pth'), subset='others',transform=test_transforms)
        return target_img, 6
    
    elif args.poison_type == 'BullseyePolytope_Bench':
        target_img, target_orig_label = get_target_poison(os.path.join(args.data_dir,f'Poisons/Transfer_Bench/bp_poisons/num_poisons={args.num_images_bp}/{target_index}'), test_transforms)
        return target_img, target_orig_label

    elif args.poison_type == 'Narcissus':
        # Target Test Sets
        test_trigger_loaders = {}
        for i in range(1,4):
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
    elif dataset == 'stl10':
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

#############
# Misc Data Utils
#############

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