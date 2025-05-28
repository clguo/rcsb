
import torch
import numpy as np
import captum
import omegaconf
from captum.attr import NoiseTunnel
from captum.attr._utils.attribution import Attribution
from torchvision.transforms import GaussianBlur
from .utils import create_grid_feature_mask

from .base import AttributionMethod


class KernelShap(AttributionMethod):

    def __init__(self, *args, **kwargs):
        super(KernelShap, self).__init__(*args, **kwargs)

    def get_attr_maps(self, x, target):
        # KernelSHAP uses feature_mask in a form of a grid
        get_attr_maps_kwargs = self.get_attr_maps_kwargs.copy()
        group_width = get_attr_maps_kwargs.pop('group_width')
        group_height = get_attr_maps_kwargs.pop('group_height')
        feature_mask = create_grid_feature_mask(
            x.shape, group_width, group_height, separate_channels=False
        )
        attr_maps = self.method.attribute(
            x, target=target, feature_mask=feature_mask,
            **get_attr_maps_kwargs, **self.noise_tunnel
        )
        torch.cuda.empty_cache()
        del feature_mask
        return attr_maps