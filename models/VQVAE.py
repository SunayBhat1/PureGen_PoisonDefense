# Ported from: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.

import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm



# Define the create_net function
def create_vqvae(net_type, num_channels=3, num_hiddens=128, num_downsampling_layers=2, num_residual_layers=2, num_residual_hiddens=32, embedding_dim=64, num_embeddings=512, use_ema=True, decay=0.99, epsilon=1e-5):
    """
    Create an EBM based on the given net_type and parameters.

    Args:
        net_type (str): The type of network to create.
        num_channels (int, optional): The number of channels in the input. Defaults to 3.
        num_hiddens (int, optional): The number of hidden units. Defaults to 128.
        num_downsampling_layers (int, optional): The number of downsampling layers. Defaults to 2.
        num_residual_layers (int, optional): The number of residual layers. Defaults to 2.
        num_residual_hiddens (int, optional): The number of hidden units in residual layers. Defaults to 32.
        embedding_dim (int, optional): The dimension of the embedding space. Defaults to 64.
        num_embeddings (int, optional): The number of embeddings. Defaults to 512.
        use_ema (bool, optional): Whether to use exponential moving average. Defaults to True.
        decay (float, optional): The decay rate for exponential moving average. Defaults to 0.99.
        epsilon (float, optional): The epsilon value for the loss term. Defaults to 1e-5.

    Returns:
        object: An instance of the created EBM network.
    """

    model_args = {
        "num_hiddens": num_hiddens,
        "num_downsampling_layers": num_downsampling_layers,
        "num_residual_layers": num_residual_layers,
        "num_residual_hiddens": num_residual_hiddens,
        "embedding_dim": embedding_dim,
        "num_embeddings": num_embeddings,
        "use_ema": use_ema,
        "decay": decay,
        "epsilon": epsilon,
    }

    if net_type == 'VQVAE':
        net = VQVAE(in_channels=num_channels, **model_args)
    else:
        raise ValueError(f"Invalid net_type: {net_type}")

    return net


class SonnetExponentialMovingAverage(nn.Module):
    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, use_ema, decay, epsilon):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_ema = use_ema
        self.decay = decay
        self.epsilon = epsilon

        limit = 3 ** 0.5
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(
            -limit, limit
        )
        if use_ema:
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)
        quantized_x = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.e_i_ts.transpose(0, 1)
        ).permute(0, 3, 1, 2)

        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()
        else:
            dictionary_loss = None

        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        quantized_x = x + (quantized_x - x).detach()

        if self.use_ema and self.training:
            with torch.no_grad():
                encoding_one_hots = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).type(flat_x.dtype)
                n_i_ts = encoding_one_hots.sum(0)
                self.N_i_ts(n_i_ts)

                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.m_i_ts(embed_sums)

                N_i_ts_sum = self.N_i_ts.average.sum()
                N_i_ts_stable = (
                    (self.N_i_ts.average + self.epsilon)
                    / (N_i_ts_sum + self.num_embeddings * self.epsilon)
                    * N_i_ts_sum
                )
                self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)

        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices.view(x.shape[0], -1),
        )



class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        layers = []
        for i in range(num_residual_layers):
            layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(num_hiddens),
                    nn.SiLU(),
                    spectral_norm(nn.Conv2d(
                        in_channels=num_hiddens,
                        out_channels=num_residual_hiddens,
                        kernel_size=3,
                        padding=1,
                    )),
                    nn.BatchNorm2d(num_residual_hiddens),
                    nn.SiLU(),
                    spectral_norm(nn.Conv2d(
                        in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1,
                    )),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = h + layer(h)
        return F.silu(h)

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_downsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super().__init__()
        conv = nn.Sequential()
        for downsampling_layer in range(num_downsampling_layers):
            if downsampling_layer == 0:
                out_channels = num_hiddens // 2
            elif downsampling_layer == 1:
                (in_channels, out_channels) = (num_hiddens // 2, num_hiddens)
            else:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            conv.add_module(
                f"down{downsampling_layer}",
                spectral_norm(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )),
            )
            conv.add_module(f"bn{downsampling_layer}", nn.BatchNorm2d(out_channels))
            conv.add_module(f"silu{downsampling_layer}", nn.SiLU())

        conv.add_module(
            "final_conv",
            spectral_norm(nn.Conv2d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                padding=1,
            )),
        )
        self.conv = conv
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )

    def forward(self, x):
        h = self.conv(x)
        return self.residual_stack(h)

class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_hiddens,
        num_upsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
        ))
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )
        upconv = nn.Sequential()
        for upsampling_layer in range(num_upsampling_layers):
            if upsampling_layer < num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)
            elif upsampling_layer == num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens // 2)
            else:
                (in_channels, out_channels) = (num_hiddens // 2, 3)

            upconv.add_module(
                f"up{upsampling_layer}",
                spectral_norm(nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )),
            )
            if upsampling_layer < num_upsampling_layers - 1:
                upconv.add_module(f"silu{upsampling_layer}", nn.SiLU())

        self.upconv = upconv

    def forward(self, x):
        h = self.conv(x)
        h = self.residual_stack(h)
        x_recon = self.upconv(h)
        return x_recon

# SonnetExponentialMovingAverage and VectorQuantizer classes remain unchanged

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_downsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
        embedding_dim,
        num_embeddings,
        use_ema,
        decay,
        epsilon,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )
        self.pre_vq_conv = spectral_norm(nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1
        ))
        self.vq = VectorQuantizer(
            embedding_dim, num_embeddings, use_ema, decay, epsilon
        )
        self.decoder = Decoder(
            embedding_dim,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )

    def quantize(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        (z_quantized, dictionary_loss, commitment_loss, encoding_indices) = self.vq(z)
        return (z_quantized, dictionary_loss, commitment_loss, encoding_indices)

    def forward(self, x):
        (z_quantized, dictionary_loss, commitment_loss, _) = self.quantize(x)
        x_recon = self.decoder(z_quantized)
        return {
            "dictionary_loss": dictionary_loss,
            "commitment_loss": commitment_loss,
            "x_recon": x_recon,
        }