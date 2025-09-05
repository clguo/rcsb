import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from omegaconf import ListConfig
import torchvision.transforms as transforms
from pathlib import Path
from functools import cache

import logging
log = logging.getLogger(__name__)


class MNISTDatasetForPreds(Dataset):
    def __init__(
            self, 
            name: str,
            split: str,
            path_data: str, 
            path_labels: str, 
            path_predictions: str, 
            filter_id: int
            ):
        super().__init__()
        assert split in ['train', 'validation']
        self.name = name
        self.path_data = Path(path_data)
        self.split = split
        self.img_size = 32

        self.filter_id = filter_id

        if len(list(self.path_data.glob("*"))) == 0:
            self.prepare_data()
        
        train = (split == 'train')
        self.mnist_dataset = MNIST(
            self.path_data,
            train = train,
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
            ])
        )

        self.length = len(self.mnist_dataset)

        self.path_predictions = Path(path_predictions)

        self.filtered_idx = np.arange(len(self.mnist_dataset))
        self.predictions_df = self.get_predictions_df()
        self.filter_data(filter_id)


    def prepare_data(self):
        MNIST(self.path_data, train=True, download=True)
        MNIST(self.path_data, train=False, download=True)

    def __len__(self):
        return self.length

    @cache
    def get_labels(self):
        return [self.mnist_dataset[i][1] for i in range(len(self.mnist_dataset))]

    
    def __getitem__(self, index):
        index = self.map_index(index)
        img, label = self.mnist_dataset[index] 

        return img, index, label,  -1
    
    def get_predictions_df(self):
        if self.path_predictions.exists():
            df = pd.read_csv(self.path_predictions, index_col = "idx")
            assert len(df) == len(self.mnist_dataset)
            return df

        df = pd.DataFrame({
            "idx": np.arange(len(self.mnist_dataset)),
            "pred_label": [None] * len(self.mnist_dataset)
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
                self.get_labels(),
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
