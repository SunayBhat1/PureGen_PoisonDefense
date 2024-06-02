import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the create_net function
def create_ebm(net_type, num_filters, num_channels = 3, patch_size=1):
    """
    Create an EBM based on the given net_type and parameters.

    Args:
        net_type (str): The type of network to create.
        num_filters (int): The number of filters to use in the network.
        num_channels (int, optional): The number of channels in the input image. Defaults to 3.
        patch_size (int, optional): The size of the patch. Defaults to 1.

    Returns:
        object: An instance of the created EBM network.
    """

    if net_type == 'EBM':
        net = EBM(n_c=num_channels, n_f=num_filters)
    elif net_type == 'EBMSNGAN32':
        net = EBMSNGAN32(nf=num_filters, patch_size=patch_size) 
    else:
        raise ValueError(f"Invalid EBM type: {net_type}")

    return net

#########################
# ## EBM ## #
#########################

class EBM(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05):
        super(EBM, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 8, 1, 4, 1, 0))

    def forward(self, x):
        return self.f(x).squeeze()

#############################
# ## SNGAN ARCHITECTURES # ##
#############################

# adapted from https://github.com/kwotsin/mimicry

class GBlock(nn.Module):
    r"""
    Residual block for generator.

    Uses bilinear (rather than nearest) interpolation, and align_corners
    set to False. This is as per how torchvision does upsampling, as seen in:
    https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        upsample (bool): If True, upsamples the input feature map.
        num_classes (int): If more than 0, uses conditional batch norm instead.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 upsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.learnable_sc = in_channels != out_channels or upsample
        self.upsample = upsample

        self.c1 = nn.Conv2d(self.in_channels,
                            self.hidden_channels,
                            3,
                            1,
                            padding=1)
        self.c2 = nn.Conv2d(self.hidden_channels,
                            self.out_channels,
                            3,
                            1,
                            padding=1)

        # match tf2 weight init
        torch.nn.init.xavier_uniform_(self.c1.weight)
        torch.nn.init.zeros_(self.c1.bias)
        torch.nn.init.xavier_uniform_(self.c2.weight)
        torch.nn.init.zeros_(self.c2.bias)

        self.b1 = eval_bn(self.in_channels)
        self.b2 = eval_bn(self.hidden_channels)

        self.activation = nn.ReLU()

        # Shortcut layer
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels,
                                  out_channels,
                                  1,
                                  1,
                                  padding=0)

            # match tf2 weight init
            torch.nn.init.xavier_uniform_(self.c_sc.weight)
            torch.nn.init.zeros_(self.c_sc.bias)

    def _upsample_conv(self, x, conv):
        r"""
        Helper function for performing convolution after upsampling.
        """
        return conv(
            F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=False))

    def _residual(self, x):
        r"""
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)

        return h

    def _shortcut(self, x):
        r"""
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self._upsample_conv(
                x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x):
        r"""
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)

class DBlock(nn.Module):
    """
    Residual block for discriminator.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        downsample (bool): If True, downsamples the input feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 downsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample

        # Build the layers
        self.c1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
        self.c2 = nn.Conv2d(self.hidden_channels, self.out_channels, 3, 1, 1)

        # match tf2 weight init
        torch.nn.init.xavier_uniform_(self.c1.weight)
        torch.nn.init.zeros_(self.c1.bias)
        torch.nn.init.xavier_uniform_(self.c2.weight)
        torch.nn.init.zeros_(self.c2.bias)

        self.activation = nn.ReLU()

        # Shortcut layer
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

            # match tf2 weight init
            torch.nn.init.xavier_uniform_(self.c_sc.weight)
            torch.nn.init.zeros_(self.c_sc.bias)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self.c_sc(x)
            return F.avg_pool2d(x, 2) if self.downsample else x

        else:
            return x

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)

class DBlockOptimized(nn.Module):
    """
    Optimized residual block for discriminator. This is used as the first residual block,
    where there is a definite downsampling involved. Follows the official SNGAN reference implementation
    in chainer.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build the layers
        self.c1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
        self.c2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        self.c_sc = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        # match tf2 weight init
        torch.nn.init.xavier_uniform_(self.c1.weight)
        torch.nn.init.zeros_(self.c1.bias)
        torch.nn.init.xavier_uniform_(self.c2.weight)
        torch.nn.init.zeros_(self.c2.bias)
        torch.nn.init.xavier_uniform_(self.c_sc.weight)
        torch.nn.init.zeros_(self.c_sc.bias)

        self.activation = nn.ReLU()

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        return self.c_sc(F.avg_pool2d(x, 2))

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)

class EBMSNGAN32(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN.

    Attributes:
        nf (int): Variable controlling discriminator feature map sizes.
    """
    def __init__(self, nf=128, patch_size=1):
        super().__init__()
        self.nf = nf
        self.patch_size = patch_size

        # Build layers
        self.block1 = DBlockOptimized(3 * (self.patch_size ** 2), self.nf)
        self.block2 = DBlock(self.nf, self.nf, downsample=True)
        self.block3 = DBlock(self.nf, self.nf, downsample=False)
        self.block4 = DBlock(self.nf, self.nf, downsample=False)
        self.l5 = nn.Linear(self.nf, 1, bias=False)
        self.activation = nn.ReLU()

        # match tf2 weight init
        torch.nn.init.xavier_uniform_(self.l5.weight)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        h = x

        if self.patch_size > 1:
            h = F.pixel_unshuffle(h, self.patch_size)

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)

        # Global average pooling
        # NOTE: unlike original repo, this uses average pooling, not sum pooling
        h = torch.mean(h, dim=(2, 3))
        output = self.l5(h)

        return output