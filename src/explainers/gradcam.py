import torch
import captum
import omegaconf
from captum.attr import NoiseTunnel, LayerAttribution
from captum.attr._utils.attribution import Attribution
from torchvision.transforms import GaussianBlur

from .base import AttributionMethod

class GradCAM(AttributionMethod):

    def __init__(self, *args, **kwargs):
        super(GradCAM, self).__init__(*args, **kwargs)

    def get_attr_maps(self, x, target):
        _, _, xdim, ydim = x.shape # bs x rgb x ...
        attr = self.method.attribute(
            x, target = target, **self.get_attr_maps_kwargs, **self.noise_tunnel
        )
        return LayerAttribution.interpolate(attr, (xdim, ydim))
