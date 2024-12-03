"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import numpy as np
import os
import math
from diffusers.optimization import get_cosine_schedule_with_warmup

# TPU support and FID score calculation
try: import torch_xla.core.xla_model as xm
except: pass
try:import pytorch_fid.fid_score as fid_score
except:pass


###################
#  General Utils  #
###################

def get_optimizer_scheduler(model, optimizer_type, lr, lr_schedule, train_iters, lr_warmup, lr_milestones, lr_decay_factor=0.1, weight_decay=1e-2,momentum=0.9):

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,momentum=momentum)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented.")

    if lr_schedule == 'multi_step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_milestones, gamma=lr_decay_factor)
    elif lr_schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup,
            num_training_steps=train_iters # (len(train_loader) * 50),
        )

    return optimizer, scheduler


#############
# EBM Utils #
#############

def ebm_purify(ebm_model,X_input,langevin_steps,langevin_temp=1e-4):
    """
    Purifies the input tensor X using the Energy-Based Model (EBM).

    Parameters:
    ebm_model (torch.nn.Module): The Energy-Based Model.
    X_input (torch.Tensor): The input tensor to be purified.
    langevin_steps (int, optional): The number of Langevin steps for the EBM. Defaults to 20.
    langevin_temp (float, optional): The temperature for the Langevin dynamics. Defaults to 1e-4.
    requires_grad (bool, optional): If True, the input tensor X is cloned and requires gradient. Defaults to True.
    save_interval (int or None, optional): If None, only the final purified image is returned. If an integer n, every nth step in the dynamic process is returned. Defaults to None.

    Returns:
    torch.Tensor: The purified tensor.
    """

    # EBM Langevin dynamics parameters
    langevin_init_noise = 0.0
    langevin_eps = 1.25e-2

    # Purify the input tensor using Langevin dynamics
    X_purify = torch.autograd.Variable(X_input.clone(), requires_grad=True)
    X_purify = X_purify + langevin_init_noise * torch.randn_like(X_purify)

    for ell in range(langevin_steps):
        energy = ebm_model(X_purify).sum() / langevin_temp
        grad = torch.autograd.grad(energy, [X_purify], create_graph=False)[0]
        X_purify.data -= ((langevin_eps ** 2) / 2) * grad
        X_purify.data += langevin_eps* torch.randn_like(grad)
        xm.mark_step()
    xm.mark_step()

    return X_purify.detach()


def ebm_purify_diffusion(ebm_model, X_input, langevin_steps, sample_steps, delta=1, langevin_temp=1e-4, gradient_noise_split=False):
    """
    Purifies the input tensor X using the Energy-Based Model (EBM) optimized for TPU.

    Parameters:
    ebm_model (torch.nn.Module): The Energy-Based Model.
    X_input (torch.Tensor): The input tensor to be purified.
    langevin_steps (int): The number of Langevin steps for the EBM.
    sample_steps (list): The list of sample steps for the EBM.
    delta (int, optional): The time difference between the samples. Defaults to 1.
    langevin_temp (float, optional): The temperature for the Langevin dynamics. Defaults to 1e-4.
    gradient_noise_split (bool, optional): If True, the gradient and noise is split. Defaults to False.
    """
    # EBM Langevin dynamics parameters
    langevin_eps = 1.25e-2

    # Purify the input tensor using Langevin dynamics
    X_purify = torch.autograd.Variable(X_input.clone(), requires_grad=True)

    # Pre-compute the sample conditions
    sample_indices = torch.tensor(sample_steps)
    sample_mask = torch.zeros(sample_indices.shape[0], langevin_steps, dtype=torch.bool)
    sample_mask[torch.arange(sample_indices.shape[0]), sample_indices] = True

    X_sample = torch.zeros_like(X_input)

    if gradient_noise_split:
        X_grad = torch.zeros_like(X_input)
        X_noise = torch.zeros_like(X_input)
    else:
        sample_indices_prev = sample_indices - delta
        sample_mask_prev = torch.zeros(sample_indices_prev.shape[0], langevin_steps, dtype=torch.bool)
        sample_mask_prev[torch.arange(sample_indices_prev.shape[0]), sample_indices_prev] = True

        X_delta = torch.zeros_like(X_input)
        X_prev = torch.zeros_like(X_input)

    for ell in range(langevin_steps):
        energy = ebm_model(X_purify).sum() / langevin_temp
        grad = torch.autograd.grad(energy, X_purify, create_graph=False)[0]
        X_purify.data -= (langevin_eps ** 2 / 2) * grad
        noise = torch.randn_like(grad)
        X_purify.data += langevin_eps * noise


        # Store gradients and noises for sampled indices if required
        if gradient_noise_split and sample_mask[:, ell].any():
            X_grad[sample_mask[:, ell]] = (langevin_eps ** 2 / 2) * grad[sample_mask[:, ell]].detach().clone()
            X_noise[sample_mask[:, ell]] = langevin_eps * noise[sample_mask[:, ell]].detach().clone()
            X_sample[sample_mask[:,ell]] = X_purify[sample_mask[:,ell]].detach().clone()

        if not gradient_noise_split and sample_mask_prev[:, ell].any():
            X_prev[sample_mask_prev[:,ell]] = X_purify[sample_mask_prev[:,ell]].detach().clone()

        if not gradient_noise_split and sample_mask[:, ell].any():
            X_sample[sample_mask[:,ell]] = X_purify[sample_mask[:,ell]].detach().clone()
            X_delta[sample_mask[:,ell]] = X_sample[sample_mask[:,ell]] - X_prev[sample_mask[:,ell]]

        xm.mark_step()  
    xm.mark_step()

    if gradient_noise_split:
        return X_purify.detach(), X_sample, X_grad, X_noise
    else:
        return X_purify.detach(), X_sample, X_delta
    

###################
# Persistent Bank #
###################

def init_persistent_bank(bank_loader,bank_size,device,img_size=32):
    '''
    Initialize the persistent bank with images from the training set.

    Parameters:
        bank_loader (torch.utils.data.DataLoader): The DataLoader for the training set.
        bank_size (int): The size of the bank.
        device (torch.device): The device to store the bank.
        img_size (int, optional): The size of the images. Defaults to 32.
        ebm_guided (bool, optional): If True, need an added history dimension. Defaults to False.

    Returns:
        dict: A dictionary containing the images, purified images, and timesteps of the bank.
    '''

    bank = {"purified_images": torch.zeros(bank_size, 3, img_size, img_size).to(device),
            "t": torch.zeros(bank_size).to(device),
            }

    for batch_num, (X,_) in enumerate(bank_loader):
        batch_size = X.shape[0]
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size

        if end_idx > bank_size:
            end_idx = bank_size

        bank["purified_images"][start_idx:end_idx] = X[:end_idx-start_idx].to(device)

        if end_idx == bank_size:
            break

    return bank

def update_persistent_bank(bank,batch):

    indices = np.random.choice(bank["purified_images"].shape[0],batch.shape[0],replace=False)

    bank["purified_images"][indices] = batch
    bank["t"][indices] = 0

    return bank

def update_persistent_history(bank, new_history, interval,full_chain =False):
    for img in range(len(bank["history"])):
        # Sample a random index from the new history
        if full_chain:
            idx = np.random.randint(1,len(new_history))
        else:
            idx = np.random.randint(len(new_history))

        # Update the history image and timestep
        bank["history"][img] = new_history[idx][img]
        bank["t_history"][img] = bank["t"][img] + interval * idx

        if full_chain:
            bank["history_previous"][img] = new_history[idx-1][img]

        xm.mark_step()

    return bank

###################
# Plot Utils      #
###################

def plot_checkpoint(diff_losses,images,images_purified,images_fixed, epoch, save_path,ebm_fid=-1,diff_fid=-1, mcmc_steps=1000, diff_steps=75):
    """
    Plots the EBM loss, gradient norm, and image samples at a given epoch

    Args:
        diff_losses (list): List of EBM losses
        images (torch.Tensor): Original images
        images_purified (torch.Tensor): EBM Purified images
        images_fixed (torch.Tensor): Diffusion Fixed images
        epoch (int): Current epoch
        save_path (str): Path to save the plot
        fid (bool): Whether to compute FID score
    """
    # Left plot with loss and grad norm
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    # Plot losses
    axs[0].plot(diff_losses[0:epoch], 'o-', label='Diff Loss', color='g')
    axs[0].set_xlabel('Epoch', fontsize=16, fontweight='bold')
    axs[0].set_ylabel('Diff MSE Loss', fontsize=16, fontweight='bold')

    # Plot images
    axs[1].imshow(torchvision.utils.make_grid(images, nrow=4, padding=2, pad_value=0).permute(1, 2, 0))
    axs[1].axis('off')
    axs[1].set_title(f'Original Images', fontsize=16, fontweight='bold')

    # Plot fixed images
    axs[2].imshow(torchvision.utils.make_grid(images_fixed, nrow=4, padding=2, pad_value=0).permute(1, 2, 0))
    axs[2].axis('off')
    axs[2].set_title(f'Diffusion Fixed\n FID: {diff_fid:.2f}', fontsize=16, fontweight='bold')

    fig.suptitle(f'Epoch {epoch} Info', fontsize=28, fontweight='bold')
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path)

    # Close figure
    plt.close()

def fid_score_calculation(images_1, images_2, device):
        
    # Create directories to save the images
    os.makedirs('tmp/images_1', exist_ok=True)
    os.makedirs('tmp/images_2', exist_ok=True)

    # Save the images
    for i, image in enumerate(images_1): torchvision.utils.save_image(image, f'tmp/images_1/image_{i}.png')
    for i, image in enumerate(images_2): torchvision.utils.save_image(image, f'tmp/images_2/image_{i}.png')

    score = fid_score.calculate_fid_given_paths(['tmp/images_1', 'tmp/images_2'], 128, device, 2048 ,4)

    # Delete the images
    os.system('rm -r tmp')

    return score
