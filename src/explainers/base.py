import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
from abc import ABC, abstractmethod
from .utils import normalize_attr
from captum.attr import NoiseTunnel
from captum.attr._utils.attribution import Attribution
from torchvision.transforms import GaussianBlur
from lightning.fabric.wrappers import _FabricModule
from classifiers.base import ClassifierBase

import inspect

import logging

log = logging.getLogger(__name__)

# for LRP
from captum.attr import LRP
from torchvision.transforms._presets import ImageClassification


def _get_layers_v2(self, model: nn.Module) -> None:
    for layer in model.children():
        if isinstance(layer, ImageClassification):
            continue
        if len(list(layer.children())) == 0:
            self.layers.append(layer)
        else:
            self._get_layers(layer)


# for LayerGradCam and derivitives
def _get_last_conv_layer(module: nn.Module) -> Optional[nn.Conv2d]:
    for child in reversed(list(module.children())):
        if isinstance(child, nn.Conv2d):
            return child
        layer = _get_last_conv_layer(child)
        if layer is not None:
            return layer
    return None


def sum_into_cell_view(attribution_map, cell_size, half_cell_view=False):
    bs, height, width = attribution_map.shape

    num_cells_y = height // cell_size
    num_cells_x = width // cell_size

    # if half_cell_offset:
    # else:
    #     offset = 0

    # Split the attribution map into cells and sum the attributions within each cell
    # Reshape the map into a 4D tensor of shape (num_cells_y, cell_size, num_cells_x, cell_size)
    cells = attribution_map.view(bs, num_cells_y, cell_size, num_cells_x, cell_size)
    # Sum the attributions within each cell along the last two dimensions (cell_size, cell_size)
    cell_sums = cells.sum(dim=(2, 4))

    if half_cell_view:
        cell_sums = (
            cell_sums.unsqueeze(2)
            .unsqueeze(4)
            .expand(-1, -1, 2, -1, 2)
            .reshape(bs, 2 * num_cells_y, 2 * num_cells_x)
        )

    return cell_sums.clone()


def expand_from_cell_view(cell_view, cell_size):
    bs, num_cells_y, num_cells_x = cell_view.shape
    # Expand the cell sums back to the original attribution map shape
    height, width = num_cells_y * cell_size, num_cells_x * cell_size
    image_view = (
        cell_view.unsqueeze(2)
        .unsqueeze(4)
        .expand(-1, -1, cell_size, -1, cell_size)
        .reshape(bs, height, width)
    )
    return image_view.clone()


class AttributionMethod(ABC, nn.Module):
    forward_module: ClassifierBase

    def __init__(
        self,
        method: Attribution,
        model: torch.nn.Module,
        noise_tunnel: None | dict,
        quantile: None | float,
        smoothing: None | dict,
        cell_size: None | int,
        cell_strided: bool,
        binarize: bool,
        scale: bool,
        area: float,
        get_attr_maps_kwargs: dict,
        sign: str,
        ignore_over_quantile: float,
    ):
        super().__init__()
        """
        method - Attribution used to produce attribution maps
        model - classifier wrapped with method
        noise_tunnel - kwargs for captum.attr.NoiseTunnel with which the method
                will be wrapped. If None, NoiseTunnel is not used.
        quantile - quantile order used in attribution map binarization. If None,
                no binarization is performed.
        smoothing - kwargs for torchvision.transforms.GaussianBlur used for 
                smoothing the mask. If None, no smoothing is performed
        sign - used in postprocessing to decide which part of the attribution map
                to choose. Can be either 'positive', 'negative' or 'absolute_value'.
        """
        assert area is None or (0.0 < area <= 1)
        assert quantile is None or (0.0 < quantile <= 1)
        assert quantile is None or area is None
        assert binarize ^ scale
        assert not (smoothing and cell_size), (
            "Smoothing and cell size cannot be used together"
        )

        self.forward_module: ClassifierBase = (
            model._forward_module if isinstance(model, _FabricModule) else model
        )

        method_args = inspect.getfullargspec(method)[0]
        method_args_values = {}
        if "forward_func" in method_args:
            method_args_values["forward_func"] = model
        else:
            method_args_values["model"] = model
        if "layer" in method_args:
            method_args_values["layer"] = _get_last_conv_layer(self.forward_module)

        try:
            self.method = method(**method_args_values)
        except TypeError:
            log.info(f"Caught exception when initializing {method}.")
            log.info("Attempting initialization without keyword arguments.")
            self.method = method()

        if isinstance(self.method, LRP):
            self.method._get_layers = _get_layers_v2.__get__(self.method, LRP)

        self.noise_tunnel = noise_tunnel if noise_tunnel else {}
        self.add_noise_tunnel(noise_tunnel)
        self.quantile = quantile
        self.smoothing = GaussianBlur(**smoothing) if smoothing else None
        self.cell_size = cell_size
        self.cell_strided = cell_strided
        self.binarize = binarize
        self.scale = scale
        self.area = area
        self.get_attr_maps_kwargs = get_attr_maps_kwargs if get_attr_maps_kwargs else {}
        self.sign = sign
        self.ignore_over_quantile = ignore_over_quantile

    @abstractmethod
    def get_attr_maps(*args, **kwargs):
        pass

    def filter_quantile(self, batch_maps):
        # change quantile to area so that we pick top number of attributions
        # with additional constraint that they overlap specific area size
        quantiles = batch_maps.flatten(start_dim=1).quantile(self.quantile, dim=1)
        batch_maps[batch_maps < quantiles.view(-1, 1, 1)] = 0.0
        return batch_maps

    def filter_area(self, batch_maps):
        total_area = batch_maps.shape[-1] ** 2
        map_area = (batch_maps > 0).flatten(start_dim=1).sum(dim=1)
        for map_idx, map in enumerate(batch_maps):
            # compute quantiles for map
            qs = torch.linspace(1.0, 0.0, 200, dtype=map.dtype)
            map_q_vals = map.quantile(qs)
            if map_area[map_idx] / total_area <= self.area:
                # take original map if it already satisfies the condition
                best_q = 0.0
            else:
                # find largest quantile that results in area
                # lower than the one provided
                for q_idx, q_val in enumerate(map_q_vals):
                    tmp_map_area = (map >= q_val).sum()
                    if tmp_map_area / total_area > self.area:
                        break

                best_q = map_q_vals[q_idx - 1]

            batch_maps[map_idx] = (map >= best_q) * map
        return batch_maps

    def postprocess_attr_maps(self, batch_maps, *args, **kwargs):
        # captum postprocessing adapted to tensors
        batch_maps = normalize_attr(batch_maps, sign=self.sign)

        if self.ignore_over_quantile is not None:
            ignore_idx = batch_maps > batch_maps.flatten(start_dim=1).quantile(
                self.ignore_over_quantile, dim=1
            )
            batch_maps[ignore_idx] = 0.0

        if self.cell_size:
            assert batch_maps.shape[-1] % self.cell_size == 0

            real_cell_size = self.cell_size
            if self.cell_strided:
                assert self.cell_size % 2 == 0, (
                    "The cell size must be even for half-cell offset"
                )
                offset = self.cell_size // 2
                real_cell_size = self.cell_size // 2

                cell_view = sum_into_cell_view(
                    batch_maps, self.cell_size, half_cell_view=True
                )
                batch_maps2 = F.pad(
                    batch_maps, (offset, offset, offset, offset)
                )  # Add padding
                strided_cell_view = sum_into_cell_view(
                    batch_maps2, self.cell_size, half_cell_view=True
                )
                strided_cell_view = strided_cell_view[:, 1:-1, 1:-1]  # Remove padding
                cell_view = (cell_view + strided_cell_view) / 2
            else:
                cell_view = sum_into_cell_view(batch_maps, self.cell_size)

            if self.quantile:
                cell_view = self.filter_quantile(cell_view)
            if self.area:
                cell_view = self.filter_area(cell_view)

            batch_maps = expand_from_cell_view(cell_view, real_cell_size)

        else:
            if self.smoothing:
                batch_maps = self.smoothing(batch_maps)
            if self.quantile:
                batch_maps = self.filter_quantile(batch_maps)
            if self.area:
                batch_maps = self.filter_area(batch_maps)

        if self.binarize:
            batch_maps = (batch_maps > 0).int()

            for map in batch_maps:
                log.info(
                    f"area={((map > 0).sum() / map.numel()).item()} for {self.area=}"
                )

        if self.scale:
            # subtracting/dividing by float requires float type
            if batch_maps.dtype == torch.int:
                batch_maps = batch_maps.float()

            # scales attributions to [0, 1]
            batch_min = batch_maps.flatten(start_dim=1).min(dim=1)[0].float()
            batch_maps -= batch_min.view(-1, 1, 1)
            batch_max = batch_maps.flatten(start_dim=1).max(dim=1)[0].float()
            batch_maps /= batch_max.view(-1, 1, 1)
            batch_maps = batch_maps.float()

        return batch_maps

    def add_noise_tunnel(self, noise_tunnel):
        if noise_tunnel:
            self.method = NoiseTunnel(self.method)

    def forward(self, x):
        pass
