
import torch
from torchvision import datasets, transforms as tr
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader, Dataset, Subset
import pytorch_fid.fid_score as fid_score
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import os
from tqdm import tqdm


#############
# General Utils
#############

# cinic10_mean = np.array([0.47889522, 0.47227842, 0.43047404])
# cinic10_std = np.array([0.24205776, 0.23828046, 0.25874835])

cifar10_mean = np.array([0.5, 0.5, 0.5])
cifar10_std = np.array([0.5, 0.5, 0.5])

def get_train_data(dataset_type, data_dir, use_random_transform=False, poisoned=False, poison_amount=500):

    ##############
    # Transforms #
    ##############

    # Adjust the randomecrop size and the mean and std for the dataset
    if dataset_type in ['cifar10', 'cifar10_BP', 'cifar10_GM', 'cifar10_45K', 'cinic10', 'cincic10_imagenet_subset']:
        random_crop_size = 32
        norm_mean = np.array([0.5, 0.5, 0.5])
        norm_std = np.array([0.5, 0.5, 0.5])
    elif dataset_type == 'mnist':
        random_crop_size = 28
        norm_mean = np.array([0.5])
        norm_std = np.array([0.5])

    transform = []

    if use_random_transform:
        transform.append(tr.RandomCrop(random_crop_size, padding=4))
        if dataset_type == 'mnist': transform.append(tr.Resize((32, 32)))
        transform.append(tr.RandomHorizontalFlip())
    
    transform.append(tr.ToTensor())
    transform.append(tr.Normalize(norm_mean, norm_std))

    transform = tr.Compose(transform)

    ##############
    # Load Data  #
    ##############

    # Not Poisoned
    if not poisoned:

        if dataset_type == 'cifar10':
            dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)

        elif dataset_type == 'cifar10_BP':
            train_list = torch.load(os.path.join(data_dir, 'CIFAR10_TRAIN_Split.pth'))['clean_train']
            dataset = SubsetOfList(train_list, transform=transform, start_idx=0, end_idx=4800)

        elif dataset_type == 'cifar10_GM':
            gm_used_indices = np.load('/Users/sunaybhat/Documents/GitHub/data/data_EBM_Defense/models/ebms/indices_gm_used.npy')
            dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
            dataset.data = np.delete(dataset.data, gm_used_indices, axis=0)
            dataset.targets = np.delete(dataset.targets, gm_used_indices, axis=0)

        elif dataset_type == 'cifar10_45K':
            dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
            dataset.data = dataset.data[0:45000]
            dataset.targets = dataset.targets[0:45000]

        elif dataset_type == 'cinic10':

            dataset = datasets.ImageFolder(os.path.join(data_dir, 'CINIC-10/train'), transform=transform)

        elif dataset_type == 'cincic10_imagenet_subset':
                
            dataset = datasets.ImageFolder(os.path.join(data_dir, 'CINIC-10/train'), transform=transform)
            cifar_idxs = [idx for idx, (path, label) in enumerate(dataset.samples) if 'cifar10' not in path]

            dataset = Subset(dataset, cifar_idxs)

        elif dataset_type == 'mnist':
            dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        else:
            raise NotImplementedError
        
        return dataset
        
    else:

        if dataset_type == 'cifar10':
        
            poison_tuples_all, poison_indices_all = get_poisoned_subset_cifar10_narcissus(data_dir, poison_amount)
            cifar = True
            
        elif dataset_type == 'cinic10':
                
            poison_tuples_all, poison_indices_all = get_poisoned_subset_cifar10_narcissus(data_dir, poison_amount, cifar=False)
            cifar = False

        else:
            raise NotImplementedError

        dataset = Poisoned_Dataset(data_dir, transform=transform, num_per_label=9000,
                                    poison_tuple_list=poison_tuples_all, poison_indices=poison_indices_all,
                                    cifar=cifar)
    
        return dataset, poison_indices_all
    
def get_test_data(dataset_type, data_dir):

    ##############
    # Transforms #
    ##############

    # Adjust the randomecrop size and the mean and std for the dataset
    if dataset_type in ['cifar10', 'cifar10_BP', 'cifar10_GM', 'cifar10_45K', 'cinic10', 'cincic10_imagenet_subset']:
        norm_mean = np.array([0.5, 0.5, 0.5])
        norm_std = np.array([0.5, 0.5, 0.5])
    elif dataset_type == 'mnist':
        norm_mean = np.array([0.5])
        norm_std = np.array([0.5])

    transform = tr.Compose([tr.ToTensor(), tr.Normalize(norm_mean, norm_std)])

    ##############
    # Load Data  #
    ##############

    if dataset_type in ['cifar10', 'cifar10_BP', 'cifar10_GM', 'cifar10_45K']:
        dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    elif dataset_type == 'cinic10':
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'CINIC-10/test'), transform=transform)

    elif dataset_type == 'cincic10_imagenet_subset':
            
            dataset = datasets.ImageFolder(os.path.join(data_dir, 'CINIC-10/test'), transform=transform)
            cifar_idxs = [idx for idx, (path, label) in enumerate(dataset.samples) if 'cifar10' not in path]

            dataset = Subset(dataset, cifar_idxs)

    elif dataset_type == 'mnist':
        dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    else:
        raise NotImplementedError
    
    return dataset

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

def plot_checkpoint(ebm_loss, grad_norm, image_samples, epoch, save_path, channels=3):
    """
    Plots the EBM loss, gradient norm, and image samples at a given epoch

    Args:
        ebm_loss (torch.tensor): EBM loss at each epoch
        grad_norm (torch.tensor): Gradient norm at each epoch
        image_samples (torch.tensor): Image samples at each epoch
        epoch (int): Epoch to plot
        save_dir (str): Directory to save the plot
        channels (int): Number of channels in the image samples
    """
    # Left plot with loss and grad norm
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    ax1 = axs[0]
    if grad_norm is not None:
        ax2 = ax1.twinx()

    line1 = ax1.plot(ebm_loss[0:epoch], 'o-', label='Train Loss', color='g')
    if grad_norm is not None:
        line2 = ax2.plot(grad_norm[0:epoch], 'o-', label='Grad Norm', color='b')

    ax1.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Train Loss', fontsize=16, fontweight='bold')
    if grad_norm is not None:
        ax2.set_ylabel('Grad Norm', fontsize=16, fontweight='bold')
        lines = line1 + line2
    else:
        lines = line1
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc=0)

    if channels == 1:
        all_images = np.block([[image_samples[i*4+j,:,:] for j in range(4)] for i in range(4)])
        all_images = (np.clip(all_images, -1., 1.) + 1) / 2  
        axs[1].imshow(all_images, cmap='gray') 
    elif channels == 3:
        all_images = np.block([[image_samples[i*4+j,:,:,:] for j in range(4)] for i in range(4)]) 
        all_images = (np.clip(all_images, -1., 1.) + 1) / 2
        axs[1].imshow(all_images.transpose(1, 2, 0))
        


    axs[1].axis('off')
    axs[1].set_title(f'Shortrun Image Samples', fontsize=16, fontweight='bold')

    fig.suptitle(f'Epoch {epoch} Info', fontsize=20, fontweight='bold')
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path)

def plot_checkpoint_gen(args, train_loss, images_target, image_samples, gen, diffusion, epoch, save_path, grad_norm=None, channels=3):
    """
    Plots the Train loss, gradient norm, and EBM & Diffusion image samples at a given epoch

    Args:
        train_loss (torch.tensor): Train loss at each epoch
        grad_norm (torch.tensor): EBM Gradient norm at each epoch
        image_samples (torch.tensor): Image samples at each epoch
        epoch (int): Epoch to plot
        save_dir (str): Directory to save the plot
        channels (int): Number of channels in the image samples
    """
    # Left plot with loss and grad norm
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    ax1 = axs[0]
    if grad_norm is not None:
        ax2 = ax1.twinx()

    line1 = ax1.plot(train_loss[0:epoch], 'o-', label='Train Loss', color='g')
    if grad_norm is not None:
        line2 = ax2.plot(grad_norm[0:epoch], 'o-', label='Grad Norm', color='b')

    ax1.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Train Loss', fontsize=16, fontweight='bold')
    if grad_norm is not None:
        ax2.set_ylabel('Grad Norm', fontsize=16, fontweight='bold')
        lines = line1 + line2
    else:
        lines = line1
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc=0)

    with torch.no_grad():
        if args.net_type == 'ddpm_unet':
            images_recon = diff_denoise(args, gen, diffusion, images_target).detach().cpu().numpy()
        elif args.net_type == 'ddpm_unet_fixer':
            images_recon = diff_denoise_fixer(args, gen, diffusion, image_samples).detach().cpu().numpy()
        else:
            raise ValueError('Invalid net type: only accept ddpm_unet or ddpm_unet_fixer')

    if channels == 1:
        all_images = np.block([[images_recon[i*4+j,:,:] for j in range(4)] for i in range(4)])
        all_images = (np.clip(all_images, -1., 1.) + 1) / 2  
        axs[1].imshow(all_images, cmap='gray') 
    elif channels == 3:
        all_images = np.block([[images_recon[i*4+j,:,:,:] for j in range(4)] for i in range(4)]) 
        all_images = (np.clip(all_images, -1., 1.) + 1) / 2
        axs[1].imshow(all_images.transpose(1, 2, 0))
        


    axs[1].axis('off')
    axs[1].set_title(f'Shortrun Image Samples', fontsize=16, fontweight='bold')

    fig.suptitle(f'Epoch {epoch} Info', fontsize=20, fontweight='bold')
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path)


#############
# EBM Training Utils
#############

def sample_data(data, data_epsilon):
    return data + data_epsilon * torch.randn_like(data)

def initialize_persistent(image_dims, persistent_size, loader, data_epsilon, device,poisoned=False, persistent_path=None):
    state_bank = torch.zeros([0] + image_dims, device=device)
    for batch, batch in enumerate(loader):
        if poisoned:
            X_batch, y_batch, idx_batch, p_batch = batch
        else:
            X_batch, y_batch = batch
        if state_bank.shape[0] < persistent_size:
            state_batch = sample_data(X_batch.to(device), data_epsilon)
            state_bank = torch.cat((state_bank, state_batch), 0)
        else:
            break
    return state_bank

# load persistent banks with updated images and rejuvenate bank with random probability
def update_persistent(X_train, image_bank, images_update, rand_inds, data_epsilon, rejuv_prob):
    # get images for rejuvenation
    images_rejuv = sample_data(X_train, data_epsilon)
    # randomly select states to be rejuvenated
    rand_unif = torch.rand(images_update.shape[0], device=images_update.device)
    rejuv_inds = (rand_unif < rejuv_prob).float().view(-1, 1, 1, 1)
    # overwrite rejuvenated states and save in bank
    images_update = (1 - rejuv_inds) * images_update + rejuv_inds * images_rejuv
    image_bank[rand_inds] = images_update
    return image_bank


# get initial mcmc states for langevin updates
def initialize_mcmc(batch_size, image_bank):
    rand_inds = torch.randperm(image_bank.shape[0], device=image_bank.device)[0:batch_size]
    return image_bank[rand_inds], rand_inds

def langevin_step(ebm,images_samp,mcmc_temp, epsilon):
    # gradient of ebm
    grad = torch.autograd.grad(ebm(images_samp).sum() / mcmc_temp, [images_samp])[0]

    # langevin update
    images_samp.data -= ((epsilon ** 2) / 2) * grad
    images_samp.data += epsilon * torch.randn_like(images_samp)

    xm.mark_step()
    return images_samp, grad

# initialize and update images with langevin dynamics to obtain samples from finite-step MCMC distribution
def sample_ebm(ebm, images_init, mcmc_steps, mcmc_temp, epsilon):

    # iterative langevin updates of MCMC samples
    images_samp = torch.autograd.Variable(images_init.clone(), requires_grad=True)
    grad_norm = torch.zeros(1).to(images_init.device)
    for ell in range(mcmc_steps):
        images_samp, grad = langevin_step(ebm, images_samp,mcmc_temp, epsilon)
        grad_norm += ((epsilon ** 2) / 2) * grad.view(grad.shape[0], -1).norm(dim=1).mean()

    return images_samp.detach(), grad_norm.squeeze() / mcmc_steps


#############
# FID Utils
#############


def fid_score_calculation(fid_loader, device, epoch, args, save_path, mcmc_steps=None, purify_steps=None, channels=3, poisoned=False, ebm = None, gen=None, diffusion=None):

    images_1 = torch.zeros([0] + args.image_dims)
    images_2 = torch.zeros([0] + args.image_dims)
    for batch_num, batch in enumerate(fid_loader):
        if poisoned:
            X_batch, y_batch, idx_batch, p_batch = batch
        else:
            X_batch, y_batch = batch
        images_data = X_batch.to(device)
        if args.ebm is not None:
            images_init = sample_data(images_data, args.data_epsilon)
            images_sample = sample_ebm(ebm, images_init, mcmc_steps, args.mcmc_temp, args.epsilon)[0]
            
            if args.net_type =='ddpm_unet_fixer':
                # images_langevin = images_sample.clone()
                with torch.no_grad():
                    images_sample = diff_denoise_fixer(args, gen, diffusion, images_sample, purify_steps)

        elif args.net_type =='ddpm_unet':
            with torch.no_grad():
                images_sample = diff_denoise(args, gen, diffusion, images_data, purify_steps)


        if batch_num == 0 and xm.get_ordinal() == 0:

            if channels == 1:
                plot_images_data = np.block([[images_data[i*4+j,:,:].detach().cpu().numpy() for j in range(4)] for i in range(4)])
                plot_images_sample = np.block([[images_sample[i*4+j,:,:].detach().cpu().numpy() for j in range(4)] for i in range(4)])
            elif channels == 3:
                plot_images_data = np.block([[images_data[i*4+j,:,:,:].detach().cpu().numpy() for j in range(4)] for i in range(4)])
                plot_images_sample = np.block([[images_sample[i*4+j,:,:,:].detach().cpu().numpy() for j in range(4)] for i in range(4)])

        if batch_num == 100:
            break

        images_data_cpu = xm.all_gather(images_data, 0).cpu()
        images_sample_cpu = xm.all_gather(images_sample, 0).cpu()
        if xm.get_ordinal() == 0:
            images_1 = torch.cat((images_1, images_data_cpu), 0)
            images_2 = torch.cat((images_2, images_sample_cpu), 0)

    if xm.get_ordinal() == 0:

        # Create directories to save the images
        os.makedirs('images_1', exist_ok=True)
        os.makedirs('images_2', exist_ok=True)

        # Save the images
        for i, image in enumerate(images_1):
            torchvision.utils.save_image(image, f'images_1/image_{i}.png')
        for i, image in enumerate(images_2):
            torchvision.utils.save_image(image, f'images_2/image_{i}.png')

        score = fid_score.calculate_fid_given_paths(['images_1', 'images_2'], 16, device, 2048 ,4)

        # Delete the images
        os.system('rm -rf images_1')
        os.system('rm -rf images_2')


        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        plot_images_data = (np.clip(plot_images_data, -1., 1.) + 1) / 2   
        if channels == 1:
            axs[0].imshow(plot_images_data.squeeze(), cmap='gray')
        elif channels == 3:
            axs[0].imshow(plot_images_data.transpose(1, 2, 0))
        axs[0].axis('off')
        axs[0].set_title(f'Data Image Samples', fontsize=16, fontweight='bold')

        plot_images_sample = (np.clip(plot_images_sample, -1., 1.) + 1) / 2   
        if channels == 1:
            axs[1].imshow(plot_images_sample.squeeze(), cmap='gray')
        elif channels == 3:
            axs[1].imshow(plot_images_sample.transpose(1, 2, 0))
        axs[1].axis('off')
        axs[1].set_title(f'Generated Image Samples', fontsize=16, fontweight='bold')
        
        if args.net_type is not None:
            fig.suptitle(f'Epoch {epoch} t-steps {purify_steps} | FID Score: {score:.2f}', fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'Epoch {epoch} Steps {mcmc_steps} | FID Score: {score:.2f}', fontsize=16, fontweight='bold')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Save figure
        plt.savefig(save_path)

    else:
        score = 0

    # Broadcast the score to all the cores
    score = xm.mesh_reduce('fid_score', round(float(score), 4), np.max)

    return score


#############
# EBM Poison Training Utils
#############

def get_poisoned_subset_cifar10_narcissus(data_dir, poison_amount, cifar=True):

    forward_transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    inverse_transform = transforms.Compose([transforms.Normalize((-1, -1, -1), (2, 2, 2)), 
                                            transforms.ToPILImage()])

    if cifar:
        base_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False)
    else:
        base_dataset = datasets.ImageFolder(os.path.join(data_dir, 'CINIC-10/train'))
        cifar_idxs = [idx for idx, (path, label) in enumerate(base_dataset.samples) if 'cifar10' in path]

    train_labels = np.array(base_dataset.targets)

    poison_tuples_all = []
    poison_indices_all = []

    for class_num in range(10):
        noise_npy = np.load(os.path.join(data_dir,f'Poisons/Narcissus/best_noise_lab{class_num}.npy'))
        best_noise = torch.from_numpy(noise_npy)

        if cifar:
            train_target_list = np.where(train_labels == class_num)[0]
        else:
            train_target_list = np.where(train_labels == class_num)[0]
            train_target_list = [idx for idx in train_target_list if idx in cifar_idxs]

        poison_indices = np.random.choice(train_target_list, poison_amount, replace=False)

        poison_tuples = [(inverse_transform(torch.clamp(forward_transform(base_dataset[i][0]) + best_noise[0], -1, 1)), class_num) for i in poison_indices]

        poison_tuples_all += poison_tuples
        poison_indices_all += list(poison_indices)
    
    return poison_tuples_all, poison_indices_all

class Poisoned_Dataset(data.Dataset):
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
            dataset = torchvision.datasets.CIFAR10(root=path, train=True, download=False)
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

#############################
### diffusion model utils ###
#############################

def diff_denoise(args, gen, diffusion, X, purify_t=None):
    if purify_t is None:
        purify_t = args.purify_t
    model_kwargs = {}
    t_start = torch.tensor(X.shape[0] * [purify_t]).long().to(device=X.device)
    x_t = diffusion.q_sample(X, t_start)
    x_0 = diffusion.ddim_sample_loop_partial(gen, x_t, purify_t, model_kwargs=model_kwargs)
    return x_0

def diff_denoise_fixer(args, gen, diffusion, x_samp, purify_t=None):
    if purify_t is None:
        purify_t = args.purify_t
    x_samp_con = x_samp.clone()
    model_kwargs = {"low_res": x_samp_con}
    x_0 = diffusion.ddim_sample_loop_partial(gen, x_samp, purify_t, model_kwargs=model_kwargs)
    return x_0

def update_gen_ddpm(args, gen, diffusion, t_sampler, gen_optim, images_target, images_samp=None, ema = None):
    if images_samp is not None:
        images_samp_con = images_samp.clone()
        # sampled images give the condition
        model_kwargs = {"low_res": images_samp_con}
    else:
        # unconditional
        model_kwargs = {}
    # random selection of timesteps
    t, _ = t_sampler.sample(images_target.shape[0], images_target.device)

    gen_optim.zero_grad()
    loss_denoise = diffusion.training_losses(gen, images_target, t, model_kwargs)
    if args.diff_output == 'start_x':
        sigmas_t = diffusion._extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, 
                                                  t, images_target.shape)
        alphas_t = diffusion._extract_into_tensor(diffusion.sqrt_alphas_cumprod, 
                                                  t, images_target.shape)
        weights = (alphas_t / sigmas_t) ** args.diff_weight_power
        weights = weights[:, 0, 0, 0]
    else:
        weights = 1.0
    loss_denoise = (weights * loss_denoise["mse"]).mean()
    loss_denoise.backward()
    xm.optimizer_step(gen_optim) 
    if args.ema:
        ema.update()

    return loss_denoise.detach().data

#############################
# ## FUNCTIONS FOR FIXER ## #
#############################

# get initial mcmc states for langevin updates
def initialize_fixer_update(args, image_bank_update, image_bank_fixed):
    rand_inds = torch.randperm(image_bank_update.shape[0], device=image_bank_update.device)[0:args.batch_size]
    return image_bank_update[rand_inds], image_bank_fixed[rand_inds], rand_inds

# load persistent banks with updated images and rejuvenate bank with random probability
def update_persistent_fixer_2(args, X_train, image_bank_update, image_bank_fixed,
                              image_bank_burnin, image_bank_fixed_burnin,
                              images_update, images_target, rand_inds,
                              images_update_burnin, images_target_burnin, rand_inds_burnin,
                              burnin_counts, burnin_counts_batch):

    # get images for rejuvenation
    images_rejuv = sample_data(X_train, args.data_epsilon)
    images_rejuv_noisy = images_rejuv + args.data_epsilon_init * torch.randn_like(images_rejuv)
    rejuv_inds = (burnin_counts_batch >= args.threshold).float()
    # update burnin counts
    burnin_counts_batch = (burnin_counts_batch + 1) * (1 - rejuv_inds)
    burnin_counts[rand_inds_burnin] = burnin_counts_batch.long()
    # update mature banks
    rejuv_inds = rejuv_inds.view(-1, 1, 1, 1)
    images_update = (1 - rejuv_inds) * images_update + rejuv_inds * images_update_burnin
    image_bank_update[rand_inds] = images_update
    images_target = (1 - rejuv_inds) * images_target + rejuv_inds * images_target_burnin
    image_bank_fixed[rand_inds] = images_target
    # update burnin banks
    images_update_burnin = (1 - rejuv_inds) * images_update_burnin + rejuv_inds * images_rejuv_noisy
    image_bank_burnin[rand_inds_burnin] = images_update_burnin
    images_target_burnin = (1 - rejuv_inds) * images_target_burnin + rejuv_inds * images_rejuv
    image_bank_fixed_burnin[rand_inds_burnin] = images_target_burnin

    return image_bank_update, image_bank_fixed, image_bank_burnin, image_bank_fixed_burnin, burnin_counts