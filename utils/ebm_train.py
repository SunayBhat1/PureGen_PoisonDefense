import torch
from torchvision import datasets, transforms
try:
    import torch_xla.core.xla_model as xm
except:
    pass
from torch.utils.data import DataLoader, Dataset, Subset
try:
    import pytorch_fid.fid_score as fid_score
except:
    pass
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import os
from tqdm import tqdm



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
    ax2 = ax1.twinx()

    line1 = ax1.plot(ebm_loss[0:epoch], 'o-', label='EBM Loss', color='g')
    line2 = ax2.plot(grad_norm[0:epoch], 'o-', label='Grad Norm', color='b')

    ax1.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax1.set_ylabel('EBM Loss', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Grad Norm', fontsize=16, fontweight='bold')
    lines = line1 + line2
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

#############
# EBM Training Utils
#############

def sample_data(data, data_epsilon):
    return data + data_epsilon * torch.randn_like(data)

def initialize_persistent(image_dims, persistent_size, loader, data_epsilon, device,poisoned=False):
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


def fid_score_calculation(ebm, fid_loader, device, epoch, args, mcmc_steps, save_path,channels=3, batch_num_break = 60):

    images_1 = torch.zeros([0] + args.image_dims)
    images_2 = torch.zeros([0] + args.image_dims)
    for batch_num, batch in enumerate(fid_loader):
        X_batch, y_batch = batch
        
        images_data = X_batch.to(device)
        images_init = sample_data(images_data, args.data_epsilon)
        images_sample = sample_ebm(ebm, images_init, mcmc_steps, args.mcmc_temp, args.epsilon)[0]

        if batch_num == 0 and xm.get_ordinal() == 0:

            if channels == 1:
                plot_images_data = np.block([[images_data[i*4+j,:,:].detach().cpu().numpy() for j in range(4)] for i in range(4)])
                plot_images_sample = np.block([[images_sample[i*4+j,:,:].detach().cpu().numpy() for j in range(4)] for i in range(4)])
            elif channels == 3:
                plot_images_data = np.block([[images_data[i*4+j,:,:,:].detach().cpu().numpy() for j in range(4)] for i in range(4)])
                plot_images_sample = np.block([[images_sample[i*4+j,:,:,:].detach().cpu().numpy() for j in range(4)] for i in range(4)])

        if batch_num == batch_num_break:
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


