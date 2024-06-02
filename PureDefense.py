import torch
import torchvision.transforms as transforms
import io
from PIL import Image
from tqdm import tqdm
import time

try: import torch_xla.core.xla_model as xm
except: pass

from utils.EBM_models import create_ebm
from utils.Diff_models import create_diff
from diffusers import UNet2DModel, DDPMScheduler, ScoreSdeVeScheduler


class PureDefense:
    def __init__(self, device, device_type = 'xla',
                 ebm_type=None,ebm_path=None,ebm_nf=128,
                 diff_type=None,diff_path=None, 
                 diff_unet_channels=None,diff_nf=64,
                 jpeg_compression=None,
                 verbose=True
                 ):
        '''
        '''

        # Store Arguments
        self.device = device
        self.device_type = device_type
        self.ebm_type = ebm_type
        self.diff_type = diff_type
        self.EBM = None
        self.DM = None
        if jpeg_compression is not None and not isinstance(jpeg_compression, int):
            raise TypeError("jpeg_compression must be None or an int")
        self.jpeg_compression = jpeg_compression

        # Normalizations for the input and output tensors
        self.forward_ebm_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.inverse_ebm_norm = transforms.Normalize((-1, -1, -1), (2, 2, 2))

        # Load the EBM and Diffusion models
        if self.ebm_type is not None: self.get_ebm(ebm_type, ebm_path, ebm_nf,verbose)
        if self.diff_type is not None: self.get_diff(diff_type, diff_path, diff_unet_channels,nf=diff_nf,verbose=verbose)
            

    def purify(self, data_loader, 
               ebm_lang_steps=100,ebm_lang_temp=1e-4,
               diff_steps=125,reverse_only=False,
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

            if self.jpeg_compression is not None:
                input = self.jpeg_compress_batch(input)

            input = self.forward_ebm_norm(input).to(self.device)

            for i in range(purify_reps):

                if self.EBM is not None and ebm_lang_steps > 0:
                    input = self.ebm_purify(input,
                            langevin_steps=ebm_lang_steps,
                            langevin_temp=ebm_lang_temp,
                        ).squeeze(0)
                    
                if self.DM is not None:
                    input = self.diff_purify(input,diff_steps,reverse_only=reverse_only)
                if self.device_type =='xla': xm.mark_step()

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

        return X_purify.detach()

    def diff_purify(self, X_input,diff_steps,reverse_only=False):
        """
        Purifies the input tensor X using the HF-DDPM model.

        Parameters:
        X_input (torch.Tensor): The original input tensor.

        Returns:
        torch.Tensor: The purified tensor.
        """

        with torch.no_grad():

            if reverse_only:
                forward_images = X_input
            else:
                forward_images = self.scheduler.add_noise(X_input,torch.randn(X_input.shape),timesteps = torch.LongTensor([diff_steps])).to(self.device)

            reverse_images = forward_images.clone()

            for i, t in enumerate(self.scheduler.timesteps[-diff_steps:]):
                residual = self.DM(reverse_images, timestep=t,return_dict=False)[0]
                reverse_images = self.scheduler.step(residual, t, reverse_images).prev_sample

                if self.device_type =='xla': xm.mark_step()
            if self.device_type =='xla': xm.mark_step()

        return reverse_images

    def jpeg_compress_batch(self, batch):
        """
        Compresses the input batch of images using JPEG compression.

        Parameters:
        batch (torch.Tensor): The input batch of images.

        Returns:
        torch.Tensor: The compressed batch of images.
        """

        # Convert the batch to PIL images
        batch = [transforms.ToPILImage()(img) for img in list(torch.unbind(batch, dim=0))]

        # Compress the images using JPEG compression
        compressed_batch = []
        for img in batch:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=self.jpeg_compression)
            img = Image.open(buffer)
            compressed_batch.append(img)

        # Convert the compressed images to tensors
        batch = [transforms.ToTensor()(img).unsqueeze(0) for img in compressed_batch]

        return torch.stack(batch)

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
        self.EBM.eval()

        if verbose:
            num_params = sum(p.numel() for p in self.EBM.parameters() if p.requires_grad)
            print(f'Loaded {ebm_type} from {ebm_path} with {num_params} parameters')
    
    def get_diff(self, diff_type, diff_path,
                 unet_channels=(32, 32, 64, 64, 128, 128),nf=64,num_res_blocks=2,
                  verbose=True): # channels=3,
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

        if diff_type == 'HF_DDPM_PRE':
            self.DM = UNet2DModel.from_pretrained('google/ddpm-cifar10-32')

            self.scheduler = DDPMScheduler.from_pretrained('google/ddpm-cifar10-32')
        else:

            # Create the Differential model
            self.DM = create_diff(diff_type, unet_channels, nf, num_res_blocks=num_res_blocks)
            self.diff_type = diff_type

            # Load the state dictionary of the Diffusion model
            state_dict = torch.load(diff_path, map_location=torch.device('cpu'))
            self.DM.load_state_dict(state_dict)

            # Scheduler for DDPM
            self.scheduler = DDPMScheduler(num_train_timesteps=1000)

        # Move the Diffusion model to the device
        self.DM = self.DM.to(self.device)
        self.DM.eval()

        if verbose:
            num_params = sum(p.numel() for p in self.DM.parameters() if p.requires_grad)    
            print(f'Loaded {diff_type} from {diff_path} with {num_params} parameters')