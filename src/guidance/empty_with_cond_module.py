from .base import Guidance
from classifiers.base import ClassifierBase

class EmptyGuidanceWithCondModule(Guidance):
    def __init__(self, model: ClassifierBase, **kwargs):
        super(EmptyGuidanceWithCondModule, self).__init__()
        self.clf = model

    def get_cond_fn(*args, **kwargs):
        return None

    def get_cond_module(self):
        return self.clf

    def log_grad_info(self, t, batch_grad, name):
        pass