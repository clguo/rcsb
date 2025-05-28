import torch
import captum
from captum.attr import LRP
import omegaconf
from captum.attr import NoiseTunnel
from captum.attr._utils.attribution import Attribution
from torchvision.transforms import GaussianBlur
from .base import AttributionMethod

class LRP(AttributionMethod):

    def __init__(self, *args, **kwargs):
        super(LRP, self).__init__(*args, **kwargs)

    def get_attr_maps(self, x, target):
        return self.method.attribute(
            x, target = target, **self.get_attr_maps_kwargs, **self.noise_tunnel)