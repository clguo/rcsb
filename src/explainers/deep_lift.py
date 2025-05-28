import torch
import captum
import omegaconf
from captum.attr import NoiseTunnel
from captum.attr._utils.attribution import Attribution
from torchvision.transforms import GaussianBlur

from .base import AttributionMethod
from classifiers import ResNet50

class DeepLift(AttributionMethod):

    def __init__(self, *args, **kwargs):
        super(DeepLift, self).__init__(*args, **kwargs)
        self.forward_module.use_functional_relu_only()

    def get_attr_maps(self, x, target):
        return self.method.attribute(
            x, target = target, **self.get_attr_maps_kwargs, **self.noise_tunnel)