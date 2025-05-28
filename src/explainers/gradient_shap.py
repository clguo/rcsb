import torch
import captum
import omegaconf
from captum.attr import NoiseTunnel
from captum.attr._utils.attribution import Attribution
from torchvision.transforms import GaussianBlur

from .base import AttributionMethod

class GradientShap(AttributionMethod):

    def __init__(self, *args, **kwargs):
        super(GradientShap, self).__init__(*args, **kwargs)

    def get_attr_maps(self, x, target):
        baselines = torch.randn((1, *x.shape[1:])).to(x.device)
        return self.method.attribute(
            x, baselines=baselines, target = target, **self.get_attr_maps_kwargs, **self.noise_tunnel)