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


cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2023, 0.1994, 0.2010)

cifar_mean_gm = (0.4914, 0.4822, 0.4465)
cifar_std_gm = (0.2471, 0.2435, 0.2616)
        

#############
# Data Loaders 
#############

def get_base_poisoned_dataset(args,poison_tuple_list, poison_indices, ebm_model, diff_model, scheduler, device, train_transform=None):

    if args.poison_mode == 'from_scratch':
        train_data_poisoned_base = Poisoned_Dataset_Base(args.data_dir,
                                                         dataset=args.dataset,
                                                         num_per_label=5000 if args.dataset == 'cifar10' else 9000,
                                                         poison_tuple_list=poison_tuple_list,
                                                         poison_indices = poison_indices,
                                                         )
    elif args.poison_mode == 'transfer':
        if args.poison_type == 'BullseyePolytope':
            train_data_poisoned_base = Poisoned_Dataset_Base(os.path.join(args.data_dir,'CIFAR10_TRAIN_Split.pth'), 
                                                    dataset=args.dataset,
                                                    num_per_label=args.num_per_class_bp,
                                                    poison_tuple_list= poison_tuple_list,
                                                    poison_indices = poison_indices,
                                                    transfer=True,
                                                    )
        elif args.poison_type == 'BullseyePolytope_Bench':
            train_data_poisoned_base = PoisonedDataset_Bench(args.data_dir,args.dataset,poison_tuple_list,
                                                        2500,poison_indices)
            
        elif args.poison_type == 'Narcissus':
            train_data_poisoned_base = Poisoned_Dataset_Base(os.path.join(args.data_dir,'CIFAR10_TRAIN_Split.pth'), 
                                                    dataset=args.dataset,
                                                    num_per_label=args.num_per_class_narcissus,
                                                    poison_tuple_list= poison_tuple_list,
                                                    poison_indices = poison_indices,
                                                    transfer=True,
                                                    )
        
    base_loader_poisoned = torch.utils.data.DataLoader(train_data_poisoned_base, batch_size=args.batch_size, shuffle=False,num_workers=4)


    train_data_poisoned = PoisonedDataset_EBM(args, 
                                            base_loader_poisoned, 
                                            ebm_model, 
                                            diff_model, 
                                            scheduler,
                                            None if args.defense == 'EBM' and args.purify_freq > 0 else train_transform, 
                                            device, 
                                            n_steps=args.pre_purify_steps if args.defense == 'EBM' and args.purify_freq > 0 else None)

    if args.model in ['HLB','ResNet18_HLB']:
        aug = {'flip': args.hlb_flip}
        if args.hlb_translate is not None: aug['translate'] = args.hlb_translate
        if args.hlb_cutout is not None: aug['cutout'] = args.hlb_cutout
        train_data_poisoned = CifarLoader(train_data_poisoned.data, train=True, batch_size=args.batch_size, aug=aug, device=device,no_poison=args.no_poison,path=args.data_dir)

    return train_data_poisoned


class PoisonedDataset_EBM(data.Dataset):

    def __init__(self, args, base_loader, ebm_model,diff_model, scheduler, transform, device, n_steps=None):
        # Extract the features
        input_list, label_list, p_list, index_list = [], [], [], []

        if args.device_type != 'xla' or (args.device_type == 'xla' and xm.is_master_ordinal()):
            if args.defense == 'EBM':
                pbar = tqdm(total=len(base_loader), desc="Pre-Processing and Purifying Data")
            else:
                pbar = tqdm(total=len(base_loader), desc="Pre-Processing Data")
        
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        if args.defense in ['EBM','EBM_Diff']:
            if n_steps is None: 
                if args.purify_freq > 0:
                    steps = args.pre_purify_steps
                else:
                    steps = args.langevin_steps
            else: 
                steps = n_steps

            forward_ebm_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            inverse_ebm_norm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
        
        # Iterate through the base loader
        for input, target, index, p in (base_loader):
            # If using EBM, purify the data
            if args.defense in ['EBM','EBM_Diff']:
                input = forward_ebm_norm(input).to(device)

                if steps > 0: 
                    input_purify = purify(input, ebm_model,
                            purify_reps=args.purify_reps,
                            reps_mode=args.purify_reps_mode,
                            langevin_steps=steps,
                            langevin_temp=args.langevin_temp,
                            device_type=args.device_type
                        ).squeeze(0)
                else:
                    input_purify = input
                    
                

                if args.ebm_perturb_clamp is None:
                    input = inverse_ebm_norm(input_purify)
                else: 
                    input = inverse_ebm_norm(input)
                    delta = inverse_ebm_norm(input_purify) - input 
                    clamped_delta = delta.clamp(-args.ebm_perturb_clamp/255, args.ebm_perturb_clamp/255) ## TODO COnfirm this!!!!
                    input = input + clamped_delta

            if args.defense in ['Diff','EBM_Diff']:
                
                sample = input.to(device)
                for t in scheduler.timesteps:
                    with torch.no_grad(): residual = diff_model(sample, t).sample
                    sample = scheduler.step(residual, t, sample).prev_sample
                    xm.mark_step()

                input = sample

            if args.purify_reps_mode == 'repeat':
                target = target.repeat(args.purify_reps)
                p = p.repeat(args.purify_reps)
                index = index.repeat(args.purify_reps)

            input_list.extend([transforms.ToPILImage()(img.squeeze(0)) for img in list(torch.unbind(input, dim=0))])
            label_list.extend(list(torch.unbind(target, dim=0)))
            p_list.extend(list(torch.unbind(p, dim=0)))
            index_list.extend(list(torch.unbind(index, dim=0)))

            if xm.is_master_ordinal():
                pbar.update(1)
            
            if args.device_type == 'xla': xm.mark_step()
                
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
    def __init__(self, path, dataset='cifar10', num_per_label=5000, 
                    poison_tuple_list=[], poison_indices=[], 
                    transfer=False,
                 ): 
        """
        Args:
            path (str): The path to the file containing the image and label data
            num_per_label (int): The number of images per label to include in the dataset
            poison_tuple_list (list): A list of tuples containing the poison images and their respective labels
            poison_indices (list): A list of the indices of the poison images in the dataset
            transfer (bool): Whether or not to load the dataset from a transfer learning file
        """

        if transfer:
            assert dataset == 'cifar10'
            self.img_label_list = torch.load(path)['others']
        else:
            if dataset == 'cifar10':
                dataset_cifar = torchvision.datasets.CIFAR10(root=path, train=True, download=(not os.path.exists(os.path.join(path, 'cifar-10-batches-py'))))
                self.img_label_list = [(img, label) for img, label in dataset_cifar]
            elif dataset == 'cinic10': 
                dataset_cinic = torchvision.datasets.ImageFolder(root=os.path.join(path, 'CINIC-10/valid'))
                self.img_label_list = [(img, label) for img, label in dataset_cinic]
            else: 
                raise Exception("Dataset {} not supported".format(dataset))
        self.class_labels = list(range(10))

        self.to_tensor = transforms.ToTensor()

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
            p = 1
            
        else:
            idx = self.valid_indices[index - len(self.poison_tuple_list)]
            img, label = self.img_label_list[idx]
        return self.to_tensor(img), label, index, p

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
   
class CifarLoader:

    def __init__(self, dataset, train=True, batch_size=500, aug=None, drop_last=None, shuffle=None, device=0,path=None,no_poison=False):

        if train:
            if no_poison:
                dset = torchvision.datasets.CIFAR10(path, train=train,  download=(not os.path.exists(os.path.join(path, 'cifar-10-batches-py'))))
                self.images = torch.tensor(dset.data)
                self.labels = torch.tensor(dset.targets)
                self.indices = torch.arange(len(self.images))
                self.p_values = torch.zeros(len(self.images))
            else:
                self.images, self.labels, self.indices, self.p_values = zip(*dataset)
                # Convert PIL images to np unit8
                self.images = [np.array(img).astype(np.uint8) for img in self.images]
                self.images = torch.tensor(np.array(self.images))
                self.labels = torch.tensor(self.labels)
                self.indices = torch.tensor(self.indices)
                self.p_values = torch.tensor(self.p_values)

        else:
            dset = torchvision.datasets.CIFAR10(root=path, train=False, download=(not os.path.exists(os.path.join(path, 'cifar-10-batches-py'))))
            self.images = torch.tensor(dset.data)
            self.labels = torch.tensor(dset.targets)
            self.indices = torch.arange(len(self.images))
            self.p_values = torch.zeros(len(self.images))

        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = transforms.Normalize(cifar_mean, cifar_std)
        self.denormalize = transforms.Normalize(
                                tuple(-mean / std for mean, std in zip(cifar_mean, cifar_std)), 
                                tuple(1 / std for std in cifar_std)
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

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

#############
# Transform Utils
#############
    
def get_train_transforms(args):
    if args.poison_mode == 'from_scratch' or args.poison_type in ['BullseyePolytope_Bench','Narcissus']:
        train_transforms = [transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()]
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

    if args.defense == 'Friendly':
        if "uniform" in args.friendly_noise_type:
            train_transforms.append(UniformNoise(eps=args.friendly_noise_eps / 255))
        if "gaussian" in args.friendly_noise_type:
            train_transforms.append(GaussianNoise(eps=args.friendly_noise_eps / 255))
        if "bernoulli" in args.friendly_noise_type:
            train_transforms.append(BernoulliNoise(eps=args.friendly_noise_eps / 255))

    if args.poison_mode == 'from_scratch':
        train_transforms.append(transforms.Normalize(cifar_mean_gm, cifar_std_gm))
    elif args.poison_mode == 'transfer':
        train_transforms.append(transforms.Normalize(cifar_mean, cifar_std))

    if args.defense == 'Friendly' or args.aug_rand_transforms:
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
    
def get_poisons(args,target_index):

    try:
        if args.poison_type == 'Gradient_Matching':
            poison_tuple_list, poison_indices, target = get_poisoned_subset_GM(os.path.join(args.data_dir,f'Poisons/Gradient_Matching/{target_index}'))

        elif args.poison_type == 'Narcissus':
            if args.poison_mode == 'from_scratch':
                if hasattr(args, 'index_list_narcissus'):
                    index_list = np.load(os.path.join(args.data_dir,'models/ebms',f'{args.index_list_narcissus}'))
                else:
                    index_list = None
                poison_tuple_list, poison_indices, target = get_poisoned_subset_narcissus(os.path.join(args.data_dir,f'Poisons/Narcissus/size={args.noise_sz_narcissus}_eps={args.noise_eps_narcissus}/best_noise_lab{target_index}.npy'), 
                                                                                            args.data_dir, args.dataset, target_index, args.num_images_narcissus, not args.random_imgs_narcissus, index_list)
            elif args.poison_mode == 'transfer':
                poison_tuple_list, poison_indices, target = get_poisoned_subset_narcissus(os.path.join(args.data_dir,f'Poisons/Narcissus/size={args.noise_sz_narcissus}_eps={args.noise_eps_narcissus}/best_noise_lab{target_index}.npy'), 
                                                                                            os.path.join(args.data_dir,'CIFAR10_TRAIN_Split.pth'), args.dataset, target_index, args.num_images_narcissus, not args.random_imgs_narcissus, index_list=None, transfer_subset=True, subset_forward_transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std)]))
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
    
    except Exception as e:
        return None, None, e

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


def get_poisoned_subset_narcissus(poisons_path, data_dir, dataset, label, poison_amount, last_n=True, index_list=None, transfer_subset=False, subset_forward_transform=None):

    if transfer_subset:
        base_dataset = torch.load(data_dir)['others']
        train_labels = np.array([label for _, label in base_dataset])
    else:
        if dataset == 'cifar10':
            base_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=(not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py'))))
        elif dataset == 'cinic10':
            base_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'CINIC-10/valid'))
        train_labels = np.array(base_dataset.targets)

    noise_npy = np.load(poisons_path)
    best_noise = torch.from_numpy(noise_npy)
    
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

        
    forward_transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    inverse_transform = transforms.Compose([transforms.Normalize((-1, -1, -1), (2, 2, 2)), 
                                            transforms.ToPILImage()])
    
    poison_indices = np.random.choice(train_target_list, poison_amount, replace=False)
    poison_tuples = [(inverse_transform(torch.clamp(apply_noise_patch(best_noise, forward_transform(base_dataset[i][0]), mode='add'), -1, 1)), 
                      label) for i in poison_indices]
    
    return poison_tuples, poison_indices, noise_npy

def get_target_narcissus(data_dir, dataset, poison, label, transform_test, multi_test=1, clean=False): 
    # noise_npy = np.load(poisons_path)
    best_noise = torch.from_numpy(poison)
    if dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=(not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py'))))
    elif dataset == 'cinic10':
        test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'CINIC-10/test'))
    test_labels = np.array(test_dataset.targets)
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
# Gradient Matching Utils
#############

def get_poisoned_subset_GM(poisons_path): 
    with open(os.path.join(poisons_path, "poisons.pickle"), "rb") as handle:
        poison_tuples = pickle.load(handle)
        # logger.info(f"{len(poison_tuples)} poisons in this trial.")
        poisoned_label = poison_tuples[0][1]
    with open(os.path.join(poisons_path, "base_indices.pickle"), "rb") as handle:
        poison_indices = pickle.load(handle)

    return poison_tuples, poison_indices, poisoned_label

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