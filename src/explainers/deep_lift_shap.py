import torch
import captum
import omegaconf
from captum.attr import NoiseTunnel
from captum.attr._utils.attribution import Attribution
from torchvision.transforms import GaussianBlur

from .base import AttributionMethod

class DeepLiftShap(AttributionMethod):

    def __init__(self, *args, **kwargs):
        super(DeepLiftShap, self).__init__(*args, **kwargs)

    def get_attr_maps(self, x, target):
        baselines = torch.randn((2, *x.shape)).to(x.device)
        return self.method.attribute(
            x, baselines=baselines, target = target, **self.get_attr_maps_kwargs, **self.noise_tunnel)