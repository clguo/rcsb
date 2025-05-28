import wandb
from torch import nn
from abc import ABC, abstractmethod

class Guidance(ABC, nn.Module):

    @abstractmethod
    def get_cond_fn(*args, **kwargs):
        '''
        Returns function which computes gradient
        wrt input of the classifier.
        '''
        pass

    @abstractmethod
    def get_cond_module(*args, **kwargs):
        '''
        Returns the underlying nn.Module so
        that it is moved to proper device together
        with inpainting module.
        '''
        pass

    def log_grad_info(self, t, batch_grad, name):
        wandb.log({
            f'guidance/{name}/t': t[0],
            f'guidance/{name}/grad_min': batch_grad[0].min(),
            f'guidance/{name}/grad_max': batch_grad[0].max(),
            f'guidance/{name}/grad_mean': batch_grad[0].mean(),
            f'guidance/{name}/grad_norm': batch_grad[0].norm(p=2).item()})