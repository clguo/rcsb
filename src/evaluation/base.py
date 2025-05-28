from abc import ABC, abstractmethod
from torch import nn

class Metric(ABC, nn.Module):

    @abstractmethod
    def compute(self, config, classifier, inputs):
        '''
        Uses classifier and inputs to compute a specific metric(s).
        '''
        pass