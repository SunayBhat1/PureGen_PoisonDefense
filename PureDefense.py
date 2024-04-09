import os
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn

try: 
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except: pass

from utils.EBM_models import create_ebm
from utils.Diff_models import create_diffusion_model
from utils.utils_purify import timestep_to_sinusoial_tensor

# from Diffusion.gaussian_diffusion import (
#     GaussianDiffusion,
#     get_named_beta_schedule,
#     ModelMeanType,
#     ModelVarType,
#     LossType)

class PureDefense:
    def __init__(self, device, device_type = 'xla',
                 ebm_type=None,ebm_path=None,ebm_nf=128,
                 diff_type=None,diff_path=None, diff_nf=128, 
                 time_emb_dim=64, num_res_blocks=2,
                #  diff_schedule='cosine', diff_train_steps=1000, diff_output='epsilon',img_sz=32,
                 verbose=True
                 ):
        '''
        '''

        self.device = device
        self.device_type = device_type
        self.ebm_type = ebm_type
        self.diff_type = diff_type
        self.EBM = None
        self.DM = None
        # self.diffusion = None

        self.forward_ebm_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.inverse_ebm_norm = transforms.Normalize((-1, -1, -1), (2, 2, 2))

        if self.ebm_type is not None:
            self.get_ebm(ebm_type, ebm_path, ebm_nf, verbose)

        if self.diff_type is not None:
            self.get_diff(diff_type, diff_path,time_emb_dim, num_res_blocks,diff_nf)
            

    def purify(self, data_loader, 
               ebm_lang_steps=100,ebm_lang_temp=1e-4,
               diff_steps=100, diff_eta=0,
               purify_reps=1,
               pbar=True):
        """
        Purifies the data in the given DataLoader using the Energy-Based Model (EBM) and or a Differential Model.

        Parameters:
        data_loader (torch.utils.data.DataLoader): The DataLoader containing the data to be purified without.
        device (torch.device): The device to use for computations.

        Returns:
        list: The list of purified data.
        """

        input_list, label_list, p_list, index_list = [], [], [], []

        if pbar and (self.device_type != 'xla' or (self.device_type == 'xla' and xm.is_master_ordinal())):
            pbar = tqdm(total=len(data_loader), desc="Purifying Data")

        for input, target in data_loader:

            input = self.forward_ebm_norm(input).to(self.device)

            for i in range(purify_reps):

                if self.EBM is not None and ebm_lang_steps > 0:
                    input = self.ebm_purify(input,
                            langevin_steps=ebm_lang_steps,
                            langevin_temp=ebm_lang_temp,
                        ).squeeze(0)
                    
                if self.DM is not None and diff_steps > 0:
                    t_s = torch.ones(input.shape[0]) * diff_steps
                    diff_predict = self.DM(input, timestep_to_sinusoial_tensor(t_s,64).to(self.device))
                    input = input + diff_predict

                if self.device_type =='xla': xm.mark_step()

            input = self.inverse_ebm_norm(input)

            input_list.extend([transforms.ToPILImage()(img.squeeze(0)) for img in list(torch.unbind(input, dim=0))])

            label_list.extend(list(torch.unbind(target, dim=0)))

            if pbar and (self.device_type != 'xla' or (self.device_type == 'xla' and xm.is_master_ordinal())):
                pbar.update(1)

        return list(zip(input_list, label_list))

    def ebm_purify(self,X_input,langevin_steps,langevin_temp=1e-4):
        """
        Purifies the input tensor X using the Energy-Based Model (EBM).

        Parameters:
        X_input (torch.Tensor): The input tensor to be purified.
        langevin_steps (int, optional): The number of Langevin steps for the EBM. Defaults to 20.
        langevin_temp (float, optional): The temperature for the Langevin dynamics. Defaults to 1e-4.
        requires_grad (bool, optional): If True, the input tensor X is cloned and requires gradient. Defaults to True.
        device_type (str, optional): The type of device to use ('xla' or other). Defaults to 'xla'.

        Returns:
        torch.Tensor: The purified tensor.
        """

        # EBM Update
        langevin_init_noise = 0.0
        langevin_eps = 1.25e-2

        # Set true for MCMC
        X_purify = torch.autograd.Variable(X_input.clone(), requires_grad=True)

        X_purify = X_purify + langevin_init_noise * torch.randn_like(X_purify)

        for ell in range(langevin_steps):
            energy = self.EBM(X_purify).sum() / langevin_temp
            grad = torch.autograd.grad(energy, [X_purify], create_graph=False)[0]
            X_purify.data -= ((langevin_eps ** 2) / 2) * grad
            X_purify.data += langevin_eps* torch.randn_like(grad)
            if self.device_type =='xla': xm.mark_step()
        if self.device_type =='xla': xm.mark_step()

        return X_purify
    

    def diff_purfiy(self, X_input, diff_steps, eta=0,requires_grad=True):

        # Set true for MCMC
        if requires_grad:
            X_purify = torch.autograd.Variable(X_input.clone(), requires_grad=True)

        if self.diff_type == 'DDPM_UNET_EBM':
            # condition augmentation
            x_samp_con = X_purify.clone()
            model_kwargs = {"low_res": x_samp_con}
        else:
            model_kwargs = {}
            
        if self.diff_type in ('HF_DDPM_UNET', 'DDPM_UNET'):
            t_start = torch.tensor(X_purify.shape[0] * [diff_steps]).long().to(device=self.device)
            x_t = self.diffusion.q_sample(X_purify, t_start)
            x_0 = self.diffusion.ddim_sample_loop_partial(self.Diff, x_t, diff_steps, model_kwargs=model_kwargs)

        elif self.diff_type in ('DDPM_UNET_EBM'):
            if diff_steps>0:
                x_0 = self.diffusion.ddim_sample_loop_partial(self.Diff, X_purify, diff_steps, model_kwargs=model_kwargs)
            else:
                x_0 = self.diffusion.ddim_sample_loop(self.Diff, X_purify.shape, model_kwargs=model_kwargs, eta = eta)  
        else:
            raise ValueError('defense model and net_type must match')     

        if self.device_type =='xla': xm.mark_step()

        return x_0

    def get_ebm(self, ebm_type, ebm_path, nf=128, verbose=True):
        """
        Loads an Energy-Based Model (EBM) from a specified path.

        Parameters:
        ebm_type (str): The name of the EBM model to be created.
        ebm_path (str): The path where the EBM model's state dictionary is stored.
        nf (int, optional): The number of filters in the EBM model. Defaults to 128.
        verbose (bool, optional): If True, prints a message confirming the successful loading of the model. Defaults to True.

        Returns:
        None. The method directly modifies the `self.EBM` attribute of the class instance.
        """

        # Create the EBM model
        self.EBM = create_ebm(ebm_type, nf)
        self.ebm_type = ebm_type

        # Load the state dictionary of the EBM model
        state_dict = torch.load(ebm_path, map_location=torch.device('cpu'))
        self.EBM.load_state_dict(state_dict)

        # Move the EBM model to the device
        self.EBM = self.EBM.to(self.device)

        if verbose: print(f'Loaded {ebm_type} from {ebm_path}')
    
    def get_diff(self, diff_type, diff_path,
                time_emb_dim=64, num_res_blocks=2,
                 nf=128, channels=3,
                #  diff_schedule, diff_steps,
                #  diff_output='epsilon',
                  verbose=True):
        """
        Loads a Differential Model from a specified path.

        Parameters:
        diff_type (str): The name of the Differential model to be created.
        diff_path (str): The path where the Differential model's state dictionary is stored.
        nf (int, optional): The number of filters in the Differential model. Defaults to 128.
        img_sz (int, optional): The size of the input images. Defaults to 32.
        verbose (bool, optional): If True, prints a message confirming the successful loading of the model. Defaults to True.

        Returns:
        None. The method directly modifies the `self.Diff` attribute of the class instance.
        """

        # Create the Differential model
        self.DM = create_diffusion_model(diff_type,channels,channels,time_emb_dim=time_emb_dim, num_res_blocks=num_res_blocks, nf=nf)
        self.diff_type = diff_type

        # Load the state dictionary of the Differential model
        state_dict = torch.load(diff_path, map_location=torch.device('cpu'))
        self.DM.load_state_dict(state_dict)

        # Move the Diffusion model to the device
        self.DM = self.DM.to(self.device)

        # betas = get_named_beta_schedule(diff_schedule,diff_steps)                            
        # self.diffusion = GaussianDiffusion(
        #                     betas=betas,
        #                     model_mean_type=(
        #                         ModelMeanType.EPSILON if diff_output == 'epsilon'
        #                         else ModelMeanType.START_X
        #                     ),
        #                     model_var_type=ModelVarType.FIXED_LARGE,
        #                     loss_type=LossType.MSE
        #                 )

        if verbose: print(f'Loaded {diff_type} from {diff_path}')