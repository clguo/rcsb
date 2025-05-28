from abc import ABC, abstractmethod
from torch import nn

class InpainterBase(ABC, nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval()

    @abstractmethod
    def inpaint(*args, **kwargs):
        pass