import json
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import sys
from omegaconf import ListConfig
from pathlib import Path

from datasets import load_dataset

import logging
log = logging.getLogger(__name__)


class ImageNetDatasetForPreds(Dataset):
    def __init__(
            self, 
            name: str,
            split: str,
            path_data: str, 
            path_labels: str, 
            filter_id: int | ListConfig,
            path_predictions: Path = None,
            ):
        super().__init__()
        assert split in ['train', 'validation']
        self.name = name
        self.split = split

        self.img_size = 256
        self.data = load_dataset('imagenet-1k', split = self.split, trust_remote_code = True)
        self.length = len(self.data)

        self.filter_id = filter_id

        self.path_predictions = path_predictions
        self.filtered_idx = np.arange(len(self.data))
        self.predictions_df = self.get_predictions_df()
        self.filter_data(filter_id)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        index = self.map_index(index)
        item = self.data[index] 
        img, label = item['image'], item['label']

        # ported from openai/improved-diffusion
        while min(*img.size) >= 2 * self.img_size:
            img = img.resize(
                tuple(x // 2 for x in img.size), resample=Image.BOX)

        scale = self.img_size / min(*img.size)
        img = img.resize(
            tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC)

        arr = np.array(img.convert("RGB"))
        crop_y = (arr.shape[0] - self.img_size) // 2
        crop_x = (arr.shape[1] - self.img_size) // 2
        arr = arr[crop_y : crop_y + self.img_size, crop_x : crop_x + self.img_size]
        arr = arr.astype(np.float32) / 255.

        img = np.transpose(arr, [2, 0, 1])

        # Returning -1 for pred_label compatibility with the rest of the code
        return torch.from_numpy(img), index, label, -1 

    def get_predictions_df(self):
        if self.path_predictions.exists():
            df = pd.read_csv(self.path_predictions, index_col = "idx")
            assert len(df) == len(self.data)
            return df

        df = pd.DataFrame({
            "idx": np.arange(len(self.data)),
            "pred_label": [None] * len(self.data)
        }).set_index("idx")
        df.to_csv(self.path_predictions, index = True)

        return df

    def save_predictions(self, data: np.ndarray):
        assert len(data) == len(self.filtered_idx)

        data = pd.DataFrame(
            data.numpy(force = True), 
            columns = ['idx', 'pred_label']
        )
        # Map from dataloaded idx to filtered idx
        data.set_index("idx", inplace = True)

        self.predictions_df.loc[
            self.filtered_idx, "pred_label"
        ] = data.loc[self.filtered_idx, "pred_label"].astype(int)

        self.predictions_df.to_csv(self.path_predictions, index = True)

    def filter_data(self, filter_id):

        if filter_id is not None:
            gt_labels = pd.DataFrame(
                self.data['label'],
                columns=['gt_label'],
            )

            if isinstance(filter_id, int):
                filtered_idx = gt_labels[
                    (gt_labels.gt_label == filter_id) &
                    self.predictions_df.pred_label.isna()
                ]
            elif isinstance(filter_id, ListConfig):
                filter_id = list(filter_id)
                assert all(isinstance(i, int) for i in filter_id)
                filtered_idx = gt_labels[
                    (gt_labels.gt_label.isin(filter_id)) &
                    self.predictions_df.pred_label.isna()
                ]
            else:
                raise ValueError(f"filter_id should be int or tuple, got {type(filter_id)}")
            filtered_idx = filtered_idx.sort_index().index.values
            self.filtered_idx = filtered_idx

            # create index mapping
            self.map_index = lambda x: int(filtered_idx[x])
            # change length
            self.length = len(filtered_idx)
        else:
            # if no filtering, index is mapped with indentity
            self.map_index = lambda x: x
