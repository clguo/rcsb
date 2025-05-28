import torch
import captum
import omegaconf
from captum.attr import NoiseTunnel
from captum.attr._utils.attribution import Attribution
from torchvision.transforms import GaussianBlur

from .base import AttributionMethod

class LIME(AttributionMethod):

    def __init__(self, *args, **kwargs):
        super(LIME, self).__init__(*args, **kwargs)

    def get_attr_maps(self, x, target):
        # LIME requires to compute a feature mask for each input
        # we insert it by using the provided feature_mask callable
        # that computes it
        get_attr_maps_kwargs = self.get_attr_maps_kwargs.copy()
        feature_mask = get_attr_maps_kwargs['feature_mask'](x)
        get_attr_maps_kwargs['feature_mask'] = feature_mask

        return self.method.attribute(
            x, target = target, **get_attr_maps_kwargs, **self.noise_tunnel)