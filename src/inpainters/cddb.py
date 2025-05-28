import torch
import numpy as np
import torch as th
import wandb
import matplotlib.pyplot as plt

import sys
sys.path.append('src/inpainters/src_cddb')

from .base import InpainterBase
from guidance import Guidance
from .src_cddb.i2sb.diffusion import Diffusion
from .src_cddb.i2sb import util

import logging
log = logging.getLogger(__name__)

class CDDB(InpainterBase):

    def __init__(self, subconfig: dict, guidance: Guidance, model_partial: torch.nn.Module, deep: bool, log_trajs: bool):
        '''
        Wrapper for CDDB that combines our abstraction with nn.Module
        for convenience (such as moving to proper device). No forward()
        is needed as all inner nn.Modules come with it setup properly and
        we put all further logic inside inpaint().

        deep - whether to use standard CDDB or CDDB deep
        subconfig - contains parameters defined by the authors
        model_partial - partially initialized nn.Module
        '''
        super().__init__()
        self.deep = deep
        self.set_config(subconfig)
        self.setup_diffusion(model_partial)
        self.setup_guidance(guidance)
        # dummy parameter to get device at any moment
        self.device_param = th.nn.Parameter(th.empty(0))
        self.log_trajs = log_trajs

    def set_config(self, subconfig):
        self.config = subconfig

    def setup_diffusion(self, model_partial):
        # setup diffusion object
        interval = self.config.interval
        betas = make_beta_schedule(
            n_timestep=interval, 
            linear_end=self.config.beta_max / interval)
        log_plot(betas, "diffusion_beta_schedule")
        
        betas = np.concatenate([betas[:interval // 2], np.flip(betas[:interval // 2])])
        log_plot(betas, "bridge_beta_schedule")

        self.diffusion = Diffusion(betas)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")
        # setup model and load checkpoint
        noise_levels = torch.linspace(self.config.t0, self.config.T, interval) * interval
        log_plot(betas, "bridge_noise_levels")
        # self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.model = model_partial(noise_levels=noise_levels)
        checkpoint = torch.load(self.config.load, map_location="cpu")
        self.model.load_state_dict(checkpoint['net'], strict = False)
        self.model.eval()
        log.info(f"[Net] Loaded network ckpt: {self.config.load}!")


    def setup_guidance(self, guidance):
        
        if guidance is not None:
            self.classifier = guidance.get_cond_module()
            self.cond_fn = guidance.get_cond_fn()

        else:
            self.classifier = None
            self.cond_fn = None


    def reverse_mask(self, x):
        return 1 - x


    def inpaint(self, x_gt: th.Tensor, x_mask: th.Tensor, guidance_classes: th.Tensor):
        '''
        x_gt - ground truth image with no mask applied
        x_mask - binary mask indicating regions to alter
        '''
        # we need x_gt to be in [-1, 1] range
        x_gt = (x_gt - 0.5) * 2
        assert x_gt.min() < 0. and x_gt.min() >= -1.

        clean_img = x_gt
        if len(x_mask.shape) < 4:
            x_mask = x_mask.unsqueeze(1)

        bs, ch, h, w = x_mask.shape
        mask = x_mask.repeat(1, ch, 1, 1)

        corrupt_img = clean_img * (1. - mask) + mask
        x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
        x1_pinv = corrupt_img
        x1_forw = corrupt_img

        cond = x1.detach() if self.config.cond_x1 else None

        assert self.config.interval is not None
        start_step = self.config.start_step or self.config.interval
        assert start_step <= self.config.interval

        nfe = self.config.nfe or start_step - 1
        steps = util.space_indices(start_step, nfe + 1)

        x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        if self.config.start_step is not None:
            x1 = self.diffusion.q_sample(start_step - 1, x_gt, x1, ot_ode=self.config.ot_ode)

        log_count = 20
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        log.info(f"[CDDB Sampling] steps={start_step}, {nfe=}, {log_steps=}!")

        def pred_x0_fn(xt, step):
            # step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
            step = torch.full((xt.shape[0],), step, dtype=torch.long).to(xt.device)
            out = self.model(xt, step, cond=cond)
            return self.compute_pred_x0(step, xt, out, clip_denoise=self.config.clip_denoise)

        def corrupt_method(img):
            # img: [-1,1]
            # img[mask==0] = img[mask==0], img[mask==1] = 1 (white)
            return img * (1. - mask) + mask, mask

        if self.deep:
            xs, pred_x0s = self.diffusion.ddpm_dps_sampling(
                                steps, 
                                pred_x0_fn, 
                                x1, 
                                x1_pinv, 
                                x1_forw,
                                x_gt=x_gt,
                                mask=mask, 
                                corrupt_type='inpaint', 
                                corrupt_method=corrupt_method, 
                                step_size=self.config.step_size,
                                ot_ode=self.config.ot_ode, 
                                log_steps=log_steps, 
                                verbose=True, 
                                results_dir=None,
                                cond_fn = self.cond_fn,
                                guidance_classes = guidance_classes)
        else:
            xs, pred_x0s = self.diffusion.dds_sampling(
                                steps, 
                                pred_x0_fn, 
                                x1, 
                                x1_pinv, 
                                x1_forw,
                                x_gt=x_gt,
                                mask=mask, 
                                corrupt_type='inpaint', 
                                corrupt_method=corrupt_method, 
                                step_size=self.config.step_size,
                                ot_ode=self.config.ot_ode, 
                                log_steps=log_steps, 
                                verbose=True, 
                                results_dir=None,
                                cond_fn = self.cond_fn,
                                guidance_classes = guidance_classes)

        b, *xdim = x1.shape
        assert xs.shape == pred_x0s.shape == (b, log_count, *xdim)
        x_inp = xs[:, 0, ...]

        if self.log_trajs:
            wandb.log({'misc/trajectory_xt': wandb.Image((xs[0] + 1) / 2)})
            wandb.log({'misc/trajectory_pred_x0': wandb.Image((pred_x0s[0] + 1) / 2)})

        # x_inp comes from [-1, 1] range
        # we scale it to [0, 1]
        x_inp = x_inp.clamp(-1, 1)
        x_inp = (x_inp / 2) + 0.5
        return x_inp
    
    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

def compute_batch(ckpt_opt, out):
    clean_img, y, mask = out
    corrupt_img = clean_img * (1. - mask) + mask
    x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
    x1_pinv = corrupt_img
    x1_forw = corrupt_img
    cond = x1.detach() if ckpt_opt.cond_x1 else None

    return corrupt_img, x1, mask, cond, y, clean_img, x1_pinv, x1_forw


def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()


def log_plot(data, name):
    fig = plt.figure()
    plt.plot(data)
    plt.grid()
    wandb.log({f'misc/{name}': wandb.Plotly.make_plot_media(fig)})
    plt.close(fig)