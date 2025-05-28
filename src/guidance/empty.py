from .base import Guidance

class EmptyGuidance(Guidance):
    def __init__(self, *args, **kwargs):
        super(EmptyGuidance, self).__init__()
        
    def get_cond_fn(*args, **kwargs):
        return None

    def get_cond_module(*args, **kwargs):
        return None

    def log_grad_info(self, t, batch_grad, name):
        pass