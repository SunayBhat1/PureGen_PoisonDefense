from abc import abstractmethod

import math

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from diffusers import UNet2DModel
from huggingface_hub import PyTorchModelHubMixin


"""
Floating point conversion.
"""

def create_diff(net_type,unet_channels,nf,num_res_blocks=4,im_sz=32,channels=3,grad_channels=1024,noise_channels=256): # was 512/32 for grad/noise channels

    if net_type == 'DM_UNET':
        net = create_hf_unet(unet_channels,num_res_blocks,im_sz,channels)
    elif net_type == 'DM_CONV':
        net = DiffConv(in_channels=channels, out_channels=channels, nf=nf)
    elif net_type == 'DM_UNET_S':
        net = SmallUNetAttention(nf=nf, num_res_blocks=num_res_blocks)
    elif net_type == 'DM_UNET_DUBHEAD':
        net = DoubleHeadedUNet(nf=nf, grad_channels=grad_channels, noise_channels=noise_channels, num_res_blocks=num_res_blocks, in_channels=channels, out_channels=channels)
    else:
        raise ValueError(f'Invalid diffusion model type: {net_type}')
    
    return net

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

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


#############################
# Double-Headed U-Net Model #
#############################

class DoubleHeadedUNet(nn.Module,PyTorchModelHubMixin):
    def __init__(self, nf=64, grad_channels=64, noise_channels=64, num_res_blocks=2, in_channels=3, out_channels=3, num_heads=4):
        super(DoubleHeadedUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.nf = nf
        self.num_heads = num_heads

        self.time_mlp = nn.Sequential(
            nn.Linear(nf, nf),
            nn.ReLU(),
            nn.Linear(nf, nf),
        )

        self.input_conv = nn.Conv2d(in_channels, nf, kernel_size=3, padding=1)
        self.down1 = self._make_res_block(nf, nf * 2, stride=2)
        self.attn1 = AttentionBlock_Small(nf * 2, num_heads)
        self.down2 = self._make_res_block(nf * 2, nf * 4, stride=2)
        self.attn2 = AttentionBlock_Small(nf * 4, num_heads)

        self.mid_block = nn.ModuleList([
            self._make_res_block(nf * 4, nf * 4) for _ in range(num_res_blocks)
        ])

        # Gradient pathway
        self.grad_up1 = self._make_res_block(nf * 4 + nf * 4, grad_channels, stride=2, transpose=True)
        self.grad_attn3 = AttentionBlock_Small(grad_channels, num_heads)
        self.grad_up2 = self._make_res_block(grad_channels + nf * 2, grad_channels // 2, stride=2, transpose=True)
        self.grad_attn4 = AttentionBlock_Small(grad_channels // 2, num_heads)

        # Noise pathway
        self.noise_up1 = self._make_res_block(nf * 4 + nf * 4, noise_channels, stride=2, transpose=True)
        self.noise_attn3 = AttentionBlock_Small(noise_channels, num_heads)
        self.noise_up2 = self._make_res_block(noise_channels + nf * 2, noise_channels // 2, stride=2, transpose=True)
        self.noise_attn4 = AttentionBlock_Small(noise_channels // 2, num_heads)

        self.grad_output_conv = nn.Conv2d(grad_channels // 2 + nf, out_channels, kernel_size=1)
        self.noise_output_conv = nn.Conv2d(noise_channels // 2 + nf, out_channels, kernel_size=1)

    def _make_res_block(self, in_channels, out_channels, stride=1, transpose=False):
        if transpose:
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1)
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        return nn.Sequential(
            conv,
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_in, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.nf))
        x = self.input_conv(x_in)
        skip_connections = []

        x = self.down1(x + t_emb.view(-1, self.nf, 1, 1))
        x = self.attn1(x)
        skip_connections.append(x)
        x = self.down2(x)
        x = self.attn2(x)
        skip_connections.append(x)

        for block in self.mid_block:
            x = block(x + t_emb.repeat(1, 4).view(-1, self.nf * 4, 1, 1))

        x_combined = torch.cat([x, skip_connections.pop()], dim=1)

        # Gradient pathway
        x_grad = self.grad_up1(x_combined)
        x_grad = self.grad_attn3(x_grad)
        x_grad = torch.cat([x_grad, skip_connections[-1]], dim=1)
        x_grad = self.grad_up2(x_grad)
        x_grad = self.grad_attn4(x_grad)
        x_grad = torch.cat([x_grad, self.input_conv(x_in)], dim=1)
        x_grad = self.grad_output_conv(x_grad)

        # # Reset skip connections for noise pathway
        # skip_connections = [self.attn1(x), self.attn2(x)]

        # Noise pathway
        x_noise = self.noise_up1(x_combined)
        x_noise = self.noise_attn3(x_noise)
        x_noise = torch.cat([x_noise, skip_connections.pop()], dim=1)
        x_noise = self.noise_up2(x_noise)
        x_noise = self.noise_attn4(x_noise)
        x_noise = torch.cat([x_noise, self.input_conv(x_in)], dim=1)
        x_noise = self.noise_output_conv(x_noise)

        return x_grad, x_noise


##########################
# Small U-Net Diff Model #
##########################

class SmallUNetAttention(nn.Module,PyTorchModelHubMixin):
    def __init__(self,nf=64,num_res_blocks=2,in_channels=3,out_channels=3, num_heads=4):
        super(SmallUNetAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.nf = nf
        self.num_heads = num_heads

        self.time_mlp = nn.Sequential(
            nn.Linear(nf, nf),
            nn.ReLU(),
            nn.Linear(nf, nf),
        )

        self.input_conv = nn.Conv2d(in_channels, nf, kernel_size=3, padding=1)
        self.down1 = self._make_res_block(nf, nf * 2, stride=2)
        self.attn1 = AttentionBlock_Small(nf * 2, num_heads)
        self.down2 = self._make_res_block(nf * 2, nf * 4, stride=2)
        self.attn2 = AttentionBlock_Small(nf * 4, num_heads)

        self.mid_block = nn.ModuleList([
            self._make_res_block(nf * 4, nf * 4) for _ in range(num_res_blocks)
        ])

        self.up1 = self._make_res_block(nf * 8, nf * 2, stride=2, transpose=True)
        self.attn3 = AttentionBlock_Small(nf * 2, num_heads)
        self.up2 = self._make_res_block(nf * 4, nf, stride=2, transpose=True)
        self.attn4 = AttentionBlock_Small(nf, num_heads)

        self.output_conv = nn.Conv2d(nf * 2, out_channels, kernel_size=1)

    def _make_res_block(self, in_channels, out_channels, stride=1, transpose=False):
        if transpose:
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1)
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        return nn.Sequential(
            conv,
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_in, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.nf))
        x = self.input_conv(x_in)
        skip_connections = []

        x = self.down1(x + t_emb.view(-1, self.nf, 1, 1))  # Add time embedding to the input
        x = self.attn1(x)
        skip_connections.append(x)
        x = self.down2(x)
        x = self.attn2(x)
        skip_connections.append(x)

        for block in self.mid_block:
            x = block(x + t_emb.repeat(1,4).view(-1, self.nf * 4, 1, 1))  # Add time embedding to the middle blocks

        x = torch.cat([x, skip_connections.pop()], dim=1)
        x = self.up1(x)
        x = self.attn3(x)
        x = torch.cat([x, skip_connections.pop()], dim=1)  # Concatenate skip connection
        x = self.up2(x)
        x = self.attn4(x)

        x = torch.cat([x, self.input_conv(x_in)], dim=1)  # Concatenate input
        x = self.output_conv(x)

        return x

class AttentionBlock_Small(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(AttentionBlock_Small, self).__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.num_heads = num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).view(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)
        k = self.k(x).view(b, self.num_heads, c // self.num_heads, h * w)
        v = self.v(x).view(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k) * (c ** (-0.5))
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        out = self.proj_out(out)

        return out

#####################
# Basic Conv Model #
#####################

class DiffConv(nn.Module,PyTorchModelHubMixin):
    def __init__(self, in_channels=3, out_channels=3, nf=64):
        super().__init__()
        self.nf = nf
        
        self.conv1 = nn.Conv2d(in_channels, nf, 3, padding=1)
        self.conv2 = nn.Conv2d(nf, nf * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(nf * 2, nf * 4, 3, padding=1)
        self.conv4 = nn.Conv2d(nf * 4, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(nf, nf),
            nn.ReLU(),
            nn.Linear(nf, nf),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.nf))
        x = self.conv1(x)
        t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        x = nn.ReLU()(x + t_emb)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        return x