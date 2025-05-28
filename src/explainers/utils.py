import torch
import warnings
from skimage.segmentation import quickshift

def _normalize_scale(attr: torch.tensor, scale_factor: float):
    if torch.any(scale_factor == 0):
        # avoiding the case where typically captum throws an error
        scale_factor[scale_factor == 0] = 1e-5
    if torch.any(scale_factor.abs() < 1e-5):
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    scale_factor = scale_factor.view(-1, 1, 1)
    assert len(attr.shape) == 3
    attr_norm = attr / scale_factor
    return torch.clip(attr_norm, -1, 1)

def _cumulative_sum_threshold(values: torch.tensor, percentile: int):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    batch_size = values.shape[0]
    sorted_vals = values.view(batch_size, -1).sort(dim = 1)[0]
    cum_sums = sorted_vals.cumsum(dim = 1)
    threshold_id = (cum_sums > cum_sums[:, -1, None] * 0.01 * percentile).int().argmax(dim = 1)
    return torch.gather(sorted_vals, 1, threshold_id[:, None]).flatten()

def normalize(x):
    '''
    Rescales x from [-1, 1] to [0, 1]
    '''
    return (x + 1.) / 2.

def normalize_attr(
        attr: torch.tensor, 
        sign: str, 
        outlier_perc: int = 2,
        reduction_axis: int = 1):

    attr_combined = attr.sum(dim = reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage
    if sign == 'all':
        threshold = _cumulative_sum_threshold(attr_combined.abs(), 100 - outlier_perc)
    elif sign == 'positive':
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif sign == 'negative':
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(attr_combined.abs(), 100 - outlier_perc)
    elif sign == 'absolute_value':
        attr_combined = attr_combined.abs()
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize sign type is not valid.")
    
    return _normalize_scale(attr_combined, threshold)


def get_superpixels(batch_imgs, quickshift_kwargs):
    masks = []
    for img in batch_imgs.permute(0, 2, 3, 1):
        mask = quickshift(img.cpu(), **quickshift_kwargs)
        masks.append(torch.from_numpy(mask))
    return torch.stack(masks).to(img.device)


def create_grid_feature_mask(input_shape: tuple, group_width: int, group_height: int, separate_channels: bool=False) -> torch.Tensor:
    """Groups pixels in a rectangle: group_width x group_height in a form of a grid.
    Returns a tensor with the shape equal to the input_shape. If separate_channels is set to true, then
    the each channel receives a separate range of feature ids in this feature mask.
    """
    assert len(input_shape) == 4
    batch_size, channels, img_width, img_height = input_shape

    assert img_width % group_width == 0
    assert img_height % group_height == 0
    num_groups_x = img_width // group_width
    num_groups_y = img_height // group_height
    num_groups = num_groups_x * num_groups_y

    grid = (
        torch.arange(num_groups)
        .reshape(num_groups_x, num_groups_y)
        .repeat_interleave(group_width, dim=0)
        .repeat_interleave(group_height, dim=1)
    )
    channel_grid = grid.reshape(1, 1, img_height, img_width).repeat(1,channels,1,1)
    if separate_channels:
        channel_offsets = (torch.arange(channels) * num_groups).reshape(1, channels, 1, 1)
        channel_grid = channel_grid + channel_offsets
    feature_mask = channel_grid.repeat_interleave(batch_size, dim=0)

    return feature_mask