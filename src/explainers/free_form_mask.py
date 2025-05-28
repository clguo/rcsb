import torch
import os
import io
import numpy as np
from itertools import cycle
import random

from .base import AttributionMethod

from logging import getLogger

logger = getLogger(__name__)


class FreeFormMask(AttributionMethod):
    def __init__(
        self, path: str, mask_size: str, mask_idx: int | None, use_random: bool, iterate: bool, *args, **kwargs
    ):
        """
        Initialize the FreeFormMask attribution method.

        Args:
            path (str): Path to the folder containing mask files.
            mask_idx (int | None): Index to select a particular mask. Mutually exclusive with use_random and iterate.
            use_random (bool): If True, sample a random mask for each image. Mutually exclusive with mask_idx and iterate.
            iterate (bool): If True, iterate over masks. Mutually exclusive with mask_idx and use_random.

        Note:
            The arguments mask_idx, use_random, and iterate are mutually exclusive. Only one should be specified.
        """
        super(FreeFormMask, self).__init__(*args, **kwargs)

        if mask_size not in (r"10-20% freeform", r"20-30% freeform", r"30-40% freeform"):
            raise ValueError(
                f"Got mask size: '{mask_size}'śś which is not one of: ('10-20% freeform', '20-30% freeform', '30-40% freeform')"
            )

        if sum([1 if mask_idx is not None else 0, 1 * use_random, 1 * iterate]) != 1:
            raise ValueError("Exactly one argument: mask_idx, use_random, iterate must be set!")

        if mask_idx is not None and (mask_idx < 0 or mask_idx >= 10_000):
            raise ValueError("Mask idx must be in range [0, 10_000)!")

        self.mask_size = mask_size
        self.mask_idx = mask_idx
        self.use_random = use_random
        self.iterate = iterate

        shape = [10000, 256, 256]

        with open(os.path.join(path, "imagenet_freeform_masks.npz"), "rb") as f:
            self.data = f.read()

        self.data = dict(np.load(io.BytesIO(self.data)))
        logger.info("Categories of masks:")
        for key in self.data:
            logger.info(key)

        for key in self.data:
            self.data[key] = np.unpackbits(self.data[key], axis=None)[: np.prod(shape)].reshape(shape).astype(np.uint8)

        self._reset()

    def get_attr_maps(self, x, target):
        batch_size = x.shape[0]
        return (
            torch.from_numpy(np.concatenate([next(self.iterator)[None, ...] for _ in range(batch_size)], axis=0)).to(
                x.device
            )
        )[None, ...]

    def _reset(self):
        if self.mask_idx is not None:
            self.iterator = cycle([self.data[self.mask_size][self.mask_idx]])
        elif self.use_random:

            def _generator():
                while True:
                    idx = random.randint(0, 10_000)
                    yield self.data[self.mask_size][idx]

            self.iterator = iter(_generator())
        elif self.iterate:
            self.iterator = cycle(self.data[self.mask_size])

    def filter_quantile(self, batch_maps):
        return batch_maps

    def filter_area(self, batch_maps):
        return batch_maps

    def postprocess_attr_maps(self, batch_maps, *args, **kwargs):
        return batch_maps
