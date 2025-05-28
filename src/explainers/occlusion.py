import torch
import captum
import omegaconf
from captum.attr import NoiseTunnel
from captum.attr._utils.attribution import Attribution
from torchvision.transforms import GaussianBlur

from .base import AttributionMethod

class Occlusion(AttributionMethod):

    def __init__(self, *args, **kwargs):
        super(Occlusion, self).__init__(*args, **kwargs)

        self.post_init()

    def post_init(self):
        # Occlusion method strictly requires tuples and hydra
        # is not currently supporting well-working tuple instantation
        self.get_attr_maps_kwargs = dict(self.get_attr_maps_kwargs)
        for k, v in self.get_attr_maps_kwargs.items():
            if isinstance(v, omegaconf.listconfig.ListConfig):
                self.get_attr_maps_kwargs[k] = tuple(v)

    def get_attr_maps(self, x, target):
        return self.method.attribute(
            x, target = target, **self.get_attr_maps_kwargs, **self.noise_tunnel)