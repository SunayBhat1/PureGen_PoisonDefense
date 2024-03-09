import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os

try: import torch_xla.core.xla_model as xm
except: pass

def ebm_update(ebm_model, X, langevin_steps , mcmc_temp, requires_grad=False, device_type='xla'):
    langevin_init_noise = 0.0
    langevin_eps = 1.25e-2

    if not X.requires_grad:
        X = torch.autograd.Variable(X.clone(), requires_grad=True)
        return_autograd_var = False
        if device_type =='xla': xm.mark_step()
    else:
        return_autograd_var = True

    X = X + langevin_init_noise * torch.randn_like(X)

    for ell in range(langevin_steps):
        energy = ebm_model(X).sum() / mcmc_temp
        grad = torch.autograd.grad(energy, [X], create_graph=requires_grad)[0]
        if requires_grad:
            X = X - ((langevin_eps ** 2) / 2) * grad
            X = X + langevin_eps* torch.randn_like(grad)
        else:
            X.data -= ((langevin_eps ** 2) / 2) * grad
            X.data += langevin_eps* torch.randn_like(grad)
        if device_type =='xla': xm.mark_step()
    if device_type =='xla': xm.mark_step()

    if not return_autograd_var:
        X = X.detach()
    return X

def purify(X, ebm_model, purify_reps=1, reps_mode='repeat', langevin_steps=20, langevin_temp=1e-4, requires_grad=True, device_type='xla'):

    batch_size = X.shape[0]

    # Repeat X for Purify Reps
    X_repeat = X.repeat([purify_reps, 1, 1, 1])

    # Set true for MCMC
    requires_grad = True
    if requires_grad:
        X_repeat = torch.autograd.Variable(X_repeat.clone(), requires_grad=True)

    X_purify = ebm_update(ebm_model, X_repeat, langevin_steps, langevin_temp, requires_grad=False, device_type=device_type)

    if device_type =='xla': xm.mark_step()

    # Avg if needed
    if reps_mode == 'mean':
        X_purify = X_purify.view(purify_reps, batch_size, X.shape[1], X.shape[2], X.shape[3])
        X_purify = torch.mean(X_purify, dim=0, keepdim=False)
    elif reps_mode == 'median':
        X_purify = X_purify.view(purify_reps, batch_size, X.shape[1], X.shape[2], X.shape[3])
        X_purify = torch.median(X_purify, dim=0, keepdim=False)[0]

    return X_purify


def purify_batch(args, input, target, p, ebm_model, device):

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    cifar_mean_gm = (0.4914, 0.4822, 0.4465)
    cifar_std_gm = (0.2471, 0.2435, 0.2616)

    forward_ebm_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    inverse_ebm_norm = transforms.Normalize((-1, -1, -1), (2, 2, 2)) 
    
    if args.poison_mode == 'from_scratch':
        inverse_norm = transforms.Normalize(mean=[-i/j for i,j in zip(cifar_mean_gm, cifar_std_gm)], std=[1/j for j in cifar_std])
        forward_norm = transforms.Normalize(mean=cifar_mean_gm, std=cifar_std_gm)
    else:
        inverse_norm = transforms.Normalize(mean=[-i/j for i,j in zip(cifar_mean, cifar_std)], std=[1/j for j in cifar_std])
        forward_norm = transforms.Normalize(mean=cifar_mean, std=cifar_std)

    input = forward_ebm_norm(inverse_norm(input))

    input = purify(input.detach(), ebm_model,
                purify_reps=args.purify_reps,
                reps_mode=args.purify_reps_mode,
                langevin_steps=args.langevin_steps,
                langevin_temp=args.langevin_temp).squeeze(0)
    
    input = forward_norm(inverse_ebm_norm(input))

    if args.purify_reps_mode == 'repeat':
        target = target.repeat(args.purify_reps)
        p = p.repeat(args.purify_reps)

    return input, target, p