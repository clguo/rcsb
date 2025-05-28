import lpips
import torch
import torch.nn.functional as F

from guidance.utils.normalization import AdaptiveNormalizer

from .base import Guidance
from .utils import ADAMGradientStabilization, l2_normalize_gradient, LossFunctionCalculator
from classifiers.base import ClassifierBase
from schedulers.base import SchedulerBase
from utils.misc import unlist

class CDDBTargetClassifierLPIPS(Guidance):

    def __init__(
            self, 
            model: ClassifierBase, 
            uses_target_clf: bool, 
            clf_scale: SchedulerBase,
            scale: SchedulerBase,
            lpips_net: str,
            lpips_scale: SchedulerBase,
            other_loss_functions: LossFunctionCalculator,
            rescale_lpips: bool,
            normalize: bool,
            adaptive_normalize: bool,
            stabilization: ADAMGradientStabilization,
            require_grad: bool,
            keep_graph: bool,
            from_tweedie: bool,
            grad_from: str,
            log: bool,
            task: str,
            adaptive_normalize_target_scale: float = 1.,):
        '''
        uses_target_clf - whether guidance utilizes target classifier
        scale - overall guidance scale
        clf_scale - classifier guidance scale
        lpips_net - LPIPS network architecture, 'vgg' or 'alex'
        lpips_scale - guidance scale for LPIPS gradients
        rescale_lpips - whether to apply inverse sigmoid to lpips before computing gradients
        normalize - whether to l2 normalize gradients
        stabilization - either None or ADAMGradientStabilization which stabilizes gradients
        require_grad - whether guidance requires computational graph of input
        keep_graph - whether to keep the computational graph after backward pass
        from_tweedie - whether to use tweedie as input or x_t
        grad_from - either "tweedie" or "x_t", indicates with respect to what the gradient
                    is computed
        log - whether to log metrics about the gradient
        task - either 'multiclass' or 'multilabel'. Determines how probabilities are obtained.
        '''
        assert grad_from in ["tweedie", "x_t"]
        super(CDDBTargetClassifierLPIPS, self).__init__()
        
        self.clf = model
        self.scale = scale
        if self.scale is not None:
            self.scale.log_scheduler_plot("guidance_scale")
        self.clf_scale = clf_scale
        if self.clf_scale is not None:
            self.clf_scale.log_scheduler_plot("clf_scale")
        self.uses_target_clf = uses_target_clf
        
        self.lpips = lpips.LPIPS(net = lpips_net) if lpips_scale is not None else None
        self.lpips_scale = lpips_scale
        if self.lpips_scale is not None:
            self.lpips_scale.log_scheduler_plot("lpips_scale")
        self.rescale_lpips = rescale_lpips

        self.other_loss_functions = other_loss_functions

        assert not (normalize and adaptive_normalize), "Cannot normalize and adaptively normalize at the same time"
        self.normalize = normalize
        self.adaptive_normalize = adaptive_normalize
        self.adaptive_normalizer = AdaptiveNormalizer(adaptive_normalize_target_scale) if adaptive_normalize else None
        self.stabilization = stabilization
        
        self.require_grad = require_grad
        self.keep_graph = keep_graph

        self.from_tweedie = from_tweedie
        self.grad_from = grad_from

        self.task = task
        self.log = log
    
    def get_cond_module(self):
        return self.clf
    
    def get_cond_fn(self):

        def cond_fn(x, t, y = None, gt = None, **kwargs):
            assert y is not None
            assert {"pred_xstart"} <= kwargs.keys()

            with torch.enable_grad():

                # use precomputed components if there are any
                if "comps" in kwargs.keys():
                    comps = kwargs["comps"]
                else:
                    comps = 0.

                # either use tweedie estimate (\hat{x}_0) or x_t as input
                if self.from_tweedie:
                    x_in = kwargs["pred_xstart"]
                else:
                    x_in = x.float()

                # compute gradient wrt x_grad, either tweedie estimate or x_t
                if self.grad_from == "tweedie":
                    x_grad = kwargs["pred_xstart"]

                elif self.grad_from == "x_t":
                    if self.from_tweedie:
                        x_grad = x
                    else:
                        x_grad = x_in

                # detach from graph if it is not required
                if not cond_fn.require_grad:
                    x_in = x_in.detach().requires_grad_(True)

                # optionally scale input to [0, 1], obtain log probs
                clf_logits = self.clf(from_m1p1_to_01(x_in))
                
                if self.task == "multiclass":
                    clf_log_probs = F.log_softmax(clf_logits, dim = -1)

                elif self.task == "multilabel":
                    clf_log_probs = F.logsigmoid(clf_logits)

                else:
                    raise NotImplementedError
                
                clf_log_probs = clf_log_probs.gather(dim = 1, index = y[:, None]).flatten()
                
                # get gradient wrt input
                comps = comps + self.clf_scale.at(unlist(t)) * clf_log_probs.sum()
                
                # get lpips values and compute gradient
                if self.lpips_scale is not None:
                    lpips_vals = self.lpips(from_01_to_m1p1(x_in), from_01_to_m1p1(gt))

                    if self.rescale_lpips:
                        # optional rescaling
                        lpips_vals = lpips_vals.log()

                    comps = comps - self.lpips_scale.at(unlist(t)) * lpips_vals.sum()
                
                # add other loss functions
                comps = comps - self.other_loss_functions.calculate_loss(x_in, gt)

                # move in negative direction to decrease lpips
                grad = torch.autograd.grad(comps, x_grad, create_graph=cond_fn.keep_graph)[0]

                with torch.no_grad():
                    
                    # optionally stabilize using adam
                    if self.stabilization is not None:
                        grad = self.stabilization(grad)

                    # optionally l2 normalize
                    if self.normalize:
                        grad = l2_normalize_gradient(grad)
                    
                    if self.adaptive_normalizer is not None:
                        grad = self.adaptive_normalizer(grad, unlist(t))

                if self.log:
                    self.log_grad_info(t, grad, 'clf')

                # lpips_scale scales relatively to gradient from classifier
                # scale scales the entire gradient
                return self.scale.at(unlist(t)) * grad
            
        cond_fn.keep_graph = self.keep_graph
        cond_fn.require_grad = self.require_grad

        return cond_fn
    
def from_m1p1_to_01(x):
    is_m1 = (x.flatten(start_dim = 1).min(dim = 1)[0] < 0.).any()
    if is_m1:
        x = x - x.flatten(start_dim = 1).min(dim = 1)[0].view(-1, 1, 1, 1)
        x = x / x.flatten(start_dim = 1).max(dim = 1)[0].view(-1, 1, 1, 1)
    return x

def from_01_to_m1p1(x):
    is_0 = (x.flatten(start_dim = 1).min(dim = 1)[0] >= 0.).any()
    if is_0:
        x = (x - 0.5) * 2
    return x