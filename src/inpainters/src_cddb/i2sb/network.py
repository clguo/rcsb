# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import pickle
import torch

from guided_diffusion_.script_util import create_model

from . import util
from .ckpt_util import (
    I2SB_IMG256_UNCOND_PKL,
    I2SB_IMG256_UNCOND_CKPT,
    I2SB_IMG256_COND_PKL,
    I2SB_IMG256_COND_CKPT,
)

import logging
log = logging.getLogger(__name__)

class Image256Net(torch.nn.Module):

    def __init__(self, noise_levels, cond, model_kwargs):
        super(Image256Net, self).__init__()

        # NOTE: for now we disable use_fp16
        self.diffusion_model = create_model(**model_kwargs)
        log.info(f"[Net] Initialized network! Size={util.count_parameters(self.diffusion_model)}!")
        self.diffusion_model.eval()
        self.cond = cond
        if cond: log.info(f"[Net] Using conditional version")
        self.register_buffer("noise_levels", noise_levels)

    def forward(self, x, steps, cond=None):

        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        x = torch.cat([x, cond], dim=1) if self.cond else x
        return self.diffusion_model(x, t)
