import torch
import captum
import omegaconf
from captum.attr import NoiseTunnel
from captum.attr._utils.attribution import Attribution
from torchvision.transforms import GaussianBlur

from .base import AttributionMethod
from .utils import create_grid_feature_mask

class FeatureAblation(AttributionMethod):

    def __init__(self, *args, **kwargs):
        super(FeatureAblation, self).__init__(*args, **kwargs)

    def get_attr_maps(self, x, target):
        # FeatureAblation uses feature_mask in a form of a grid
        get_attr_maps_kwargs = self.get_attr_maps_kwargs.copy()
        group_width = get_attr_maps_kwargs.pop('group_width')
        group_height = get_attr_maps_kwargs.pop('group_height')
        feature_mask = create_grid_feature_mask(
            x.shape, group_width, group_height, separate_channels=True
        )

        return self.method.attribute(
            x, target = target, feature_mask=feature_mask,
            **get_attr_maps_kwargs, **self.noise_tunnel
        )