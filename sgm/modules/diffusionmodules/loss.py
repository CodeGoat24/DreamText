from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
from torchvision.utils import save_image
from ...util import append_dims, instantiate_from_config
from PIL import Image, ImageDraw, ImageFont
from os.path import join as ospj
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch, *args, **kwarg):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(input.shape[0], device=input.device), input.ndim
            )
        noised_input = input + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(denoiser.w(sigmas), input.ndim)

        loss = self.get_diff_loss(model_output, input, w)
        loss = loss.mean()
        loss_dict = {"loss": loss}

        return loss, loss_dict

    def get_diff_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        
    def get_masked_diff_loss(self, model_output, target, seg_map, seg_mask, w):
        if self.type == "l2":
            seg_map = seg_map.unsqueeze(2)

            squared_diff = (w * (model_output - target) ** 2).unsqueeze(1).repeat(1, seg_map.shape[1], 1, 1, 1)

            squared_diff = squared_diff * seg_map

            squared_diff = squared_diff.view(squared_diff.size(0), squared_diff.size(1), -1)

            loss_per_seg_map = torch.mean(squared_diff, dim=-1)
            
            loss_per_seg_map = loss_per_seg_map * seg_mask

            masked_diff_loss = loss_per_seg_map.sum(dim=-1) / seg_mask.sum(dim=-1)

            return masked_diff_loss

        elif self.type == "l1":
            seg_map = seg_map.unsqueeze(2)

            squared_diff = ((model_output - target).abs()).unsqueeze(1).repeat(1, seg_map.shape[1], 1, 1, 1)

            squared_diff = squared_diff * seg_map

            squared_diff = squared_diff.view(squared_diff.size(0), squared_diff.size(1), -1)

            loss_per_seg_map = torch.mean(squared_diff, dim=-1)

            loss_per_seg_map = loss_per_seg_map * seg_mask

            masked_diff_loss = loss_per_seg_map.sum(dim=-1) / seg_mask.sum(dim=-1)

            return masked_diff_loss

class FullLoss(StandardDiffusionLoss):

    def __init__(
        self,
        seq_len=12,
        kernel_size=3,
        gaussian_sigma=0.5,
        min_attn_size=16,
        lambda_cross_loss=0.0,
        lambda_clip_loss=0.0,
        lambda_masked_loss=0.0,
        lambda_seg_loss=0.0,
        predictor_config = None,
        *args, **kwarg
    ):
        super().__init__(*args, **kwarg)

        self.gaussian_kernel_size = kernel_size
        gaussian_kernel = self.get_gaussian_kernel(kernel_size=self.gaussian_kernel_size, sigma=gaussian_sigma, out_channels=seq_len)
        self.register_buffer("g_kernel", gaussian_kernel.requires_grad_(False))

        self.min_attn_size = min_attn_size
        self.lambda_cross_loss = lambda_cross_loss
        self.lambda_masked_loss = lambda_masked_loss
        self.lambda_clip_loss = lambda_clip_loss
        self.lambda_seg_loss = lambda_seg_loss
    
    def get_gaussian_kernel(self, kernel_size=3, sigma=1, out_channels=3):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*torch.pi*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.tile(out_channels, 1, 1, 1)
        
        return gaussian_kernel

    def __call__(self, network, denoiser, conditioner, input, batch, first_stage_model, scaler):

        cond = conditioner(batch)

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(input.shape[0], device=input.device), input.ndim
            )

        noised_input = input + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(network, noised_input, sigmas, cond) # b c h w
        w = append_dims(denoiser.w(sigmas), input.ndim)

        
        diff_loss = self.get_diff_loss(model_output, input, w)
        if 'seg' in batch:
            cross_loss, seg_loss, seg_map = self.get_cross_loss(network.diffusion_model.attn_map_cache, batch["seg"], batch["seg_mask"])
        else:
            cross_loss, seg_loss, seg_map = self.get_cross_loss(network.diffusion_model.attn_map_cache, None, batch["seg_mask"])
        diff_loss = diff_loss.mean()
        cross_loss = cross_loss.mean()


        masked_diff_loss = self.get_masked_diff_loss(model_output, input, seg_map,  batch["seg_mask"], w)
        masked_diff_loss = masked_diff_loss.mean()


        loss = diff_loss + self.lambda_cross_loss * cross_loss + self.lambda_masked_loss * masked_diff_loss
        if seg_loss is not None:
            loss = loss + self.lambda_seg_loss * seg_loss.mean()

        if 'clip_loss' in cond:
            loss += self.lambda_clip_loss * cond['clip_loss'].mean()
        

        loss_dict = {
            "loss/diff_loss": diff_loss,
            "loss/cross_loss": cross_loss,
            "loss/masked_loss": masked_diff_loss,
            "loss/full_loss": loss
        }
        
        if seg_loss is not None:
            loss_dict["loss/seg_loss"] = seg_loss

        if 'clip_loss' in cond:
            loss_dict["loss/clip_loss"] = cond['clip_loss']

        return loss, loss_dict
    

    def get_cross_loss(self, attn_map_cache, seg, seg_mask):
        seg_l = seg_mask.shape[1]
        attn_maps = []
        for item in attn_map_cache:
            name = item["name"]
            if not name.endswith("t_attn"): continue
            size = item["size"]
            if size < self.min_attn_size: continue

            heads = item["heads"]

            attn_map = item['attn_map']
            bh, n, l = attn_map.shape
            attn_map = attn_map.reshape(-1, heads, n, l)
            attn_map = attn_map.permute(0, 1, 3, 2) # b, h, l, n
            attn_map = attn_map.mean(dim = 1) # b, l, n
            attn_map = attn_map.reshape((-1, l, size, size)) # b, l, s, s

            attn_map = F.interpolate(attn_map, (64, 64))

            attn_map = F.conv2d(attn_map, self.g_kernel, padding = self.gaussian_kernel_size//2, groups=seg_l) # gaussian blur on each channel

            attn_maps.append(attn_map)

        attn_map = torch.stack(attn_maps, dim=0)
        attn_map = torch.mean(attn_map, dim=0)
        

        mean_value = torch.mean(attn_map, dim=(2, 3)) 
        std_dev = torch.std(attn_map, dim=(2, 3)) 

        threshold = mean_value + 2 * std_dev 

        seg_map = attn_map > threshold.unsqueeze(-1).unsqueeze(-1) 

        seg_map = seg_map.int()
        
        attn_map = attn_map.reshape((-1, seg_l, n)) # b, l, n

        seg_map_reshaped = seg_map.reshape((-1, seg_l, n)) # b, l, n

        if seg is None:
            seg_loss = None
        else:
            seg_map_gt = F.interpolate(seg, (64, 64))

            seg_map_gt_reshaped = seg_map_gt.reshape((-1, seg_l, n)) # b, l, n

            seg_loss = abs((seg_map_reshaped - seg_map_gt_reshaped)).mean(dim=-1) * seg_mask # b l
            
            seg_loss = seg_loss.sum(dim=-1) / seg_mask.sum(dim = -1)

        n_seg_map = 1 - seg_map_reshaped

        p_loss = (seg_map_reshaped * attn_map).max(dim = -1)[0] # b, l
        n_loss = (n_seg_map * attn_map).max(dim = -1)[0] # b, l

        p_loss = p_loss * seg_mask # b, l
        n_loss = n_loss * seg_mask # b, l

        p_loss = p_loss.sum(dim = -1) / seg_mask.sum(dim = -1) # b,
        n_loss = n_loss.sum(dim = -1) / seg_mask.sum(dim = -1) # b,

        loss = n_loss - p_loss # b,


        return loss, seg_loss, seg_map
    
    