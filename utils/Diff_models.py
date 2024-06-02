from abc import abstractmethod

import math

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from diffusers import UNet2DModel

"""
Floating point conversion.
"""

def create_diff(net_type,unet_channels,nf,num_res_blocks=2,im_sz=32,channels=3):

    if net_type == 'DM_UNET':
        net = create_hf_unet(unet_channels,num_res_blocks,im_sz,channels)
    else:
        raise ValueError(f'Invalid diffusion model type: {net_type}')
    
    return net

###########################
# HuggingFace U-Net Model #
###########################

def create_hf_unet(unet_channels,num_res_blocks,im_sz,channels):

    # Create Diffusion Model
    diff_model = UNet2DModel(
        sample_size=im_sz,  # the target image resolution
        in_channels=channels,  # the number of input channels, 3 for RGB images
        out_channels=channels,  # the number of output channels
        layers_per_block=num_res_blocks,  # how many ResNet layers to use per UNet block
        # block_out_channels=(64, 64, 128, 128, 256, 256),  # the number of output channels for each UNet block
        block_out_channels=unet_channels,  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    return diff_model
