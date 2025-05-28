import torch

from .base import AttributionMethod

class Random(AttributionMethod):

    def __init__(self, *args, **kwargs):
        super(Random, self).__init__(*args, **kwargs)

    def get_attr_maps(self, x, target):
        return torch.randn_like(x).clamp(-1, 1)