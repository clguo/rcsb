import torch
from .base import AttributionMethod

class EmptyExplainer(AttributionMethod):

    def __init__(self, *args, **kwargs):
        super(EmptyExplainer, self).__init__(*args, **kwargs)

    def get_attr_maps(self, x, target):
        return torch.empty_like(x)