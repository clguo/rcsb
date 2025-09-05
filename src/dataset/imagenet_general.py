import json
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import sys

from datasets import load_dataset

import logging
log = logging.getLogger(__name__)


class ImageNetGeneralDataset(Dataset):
    path_hier_json = 'data/metadata/imagenet/wordnet_hierarchy.json'

    def __init__(
            self, 
            name: str,
            split: str,
            path_data: str, 
            path_labels: str, 
            path_predictions: str, 
            n_samples: int, 
            filter_id_gt: int,
            guide_id: int,
            correct_pred: bool,
            n_skip: int):
        super().__init__()
        assert split in ['train', 'validation']
        self.name = name
        self.split = split

        self.img_size = 256
        self.n_samples = n_samples
        self.data = load_dataset('imagenet-1k', split = self.split, trust_remote_code = True)
        self.predictions = pd.read_csv(path_predictions, index_col = "idx") if path_predictions else None
        self.length = min(len(self.data), n_samples)
        self.hier = self.get_hier()
        self.filter_id_gt = filter_id_gt
        self.guide_id = guide_id
        self.correct_pred = correct_pred
        self.n_skip = n_skip

        self.filter_data(filter_id_gt, n_skip)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        index = self.map_index(index)
        item = self.data[index] 
        img, label = item['image'], item['label']

        if isinstance(self.predictions, pd.DataFrame):
            pred_label = self.predictions.loc[index]
        else:
            pred_label = -1

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
        log.info(f"Initial prediction: {pred_label.item()}")
        return torch.from_numpy(img), index, label, pred_label.item()


    def filter_data(self, filter_id_gt, n_skip):
        # if not filtering, simply pick all data with n_skip skipped samples
        if filter_id_gt is not None:

            # create df with gt labels
            gt_labels = pd.DataFrame(
                self.data['label'],
                columns=['gt_label'],
            )

            # join predicted with ground truth and pick with desired gt
            joint_labels = gt_labels.join(self.predictions)
            joint_labels = joint_labels[joint_labels.gt_label == filter_id_gt]

            # if correct_pred, pick images where predictions agree with ground truth
            if self.correct_pred:
                joint_labels = joint_labels[joint_labels.gt_label == joint_labels.pred_label]

            # else pick those that agree only with ground truth
            else:
                joint_labels = joint_labels[joint_labels.gt_label != joint_labels.pred_label]

            # in any case, skip images where the prediction already agrees with guide_id
            joint_labels = joint_labels[joint_labels.pred_label != self.guide_id]

            # get filtered index
            filtered_idx = joint_labels.sort_index().index.values

            # create index mapping
            self.map_index = lambda x: int(filtered_idx[x + n_skip])
            
            # change length
            self.length = min(len(filtered_idx), self.n_samples)

        else:
            # if no filtering, index is mapped with indentity + number of skipped
            self.map_index = lambda x: x + n_skip

    def get_hier(self):
        with open(self.path_hier_json, 'r') as f:
            hier = json.load(f)
        return hier

    def get_guidance_classes(self, config, fabric, batch_labels, batch_pred_labels):
        
        n_inp = config.exp.n_inpaints
        guidance_classes = []

        for label in batch_labels:

            if config.exp.guide_id is not None:
                guidance_classes.append([config.exp.guide_id] * n_inp)

            else:

                def _get_path_to_label(root, target_value, path = []):
                    
                    if 'children' not in root.keys():
                        if root['index'] == target_value:
                            return root, path + [root['name']]
                        else:
                            return None, None

                    for child in root['children']:
                        leaf, leaf_path = _get_path_to_label(child, target_value, path + [root['name']])

                        if leaf:
                            return leaf, leaf_path
                        
                    return None, None
                
                def _get_leaf_values(node):
                    if 'children' not in node.keys():
                        return [node['index']]
                    
                    leaf_values = []
                    for child in node['children']:
                        leaf_values.extend(_get_leaf_values(child))
                    
                    return leaf_values

                def _get_neighborhood_labels(hier, path_to_label):

                    if len(path_to_label) <= 1:
                        # if only hierarchy id, then just return hier
                        # with synthetic children
                        pass
                    else:
                        # pop first name since it is the hierarchy id
                        path_to_label.pop(0)
                        while len(path_to_label) > 1:
                            # pop first item from path
                            name = path_to_label.pop(0)
                            for child_id, child in enumerate(hier['children']):
                                # search for that name in children of current root
                                if name == child['name']:
                                    # if found, go one level down
                                    hier = hier['children'][child_id]
                                    break
                        
                    # extract children which has the last name from path_to_label
                    try:
                        root_list = [e for e in hier['children'] if path_to_label[0] == e['name']]
                    except:
                        import pdb; pdb.set_trace()

                    if len(root_list) == 0:
                        # it may happen that it is not found, then get all leaves
                        leaf_values = _get_leaf_values(hier)
                    else:
                        root = root_list[0]
                        leaf_values = _get_leaf_values(root)
                        
                    return list(set(leaf_values))

                leaf, path_to_label = _get_path_to_label(self.hier, label)
                assert leaf['index'] == label

                depth_inc = 0
                n_neighs = 1
                # we iteratively increase depth just in case
                # only label was found as leaf
                while n_neighs == 1:
                    if len(path_to_label[:(-config.exp.guide_depth - depth_inc)]) == 0:
                        # if even with increasing depth no neighbors were found
                        # just use all leaves
                        neigh_labels = _get_neighborhood_labels(self.hier, path_to_label[:1])
                        break
                    neigh_labels = _get_neighborhood_labels(
                        self.hier, path_to_label[:(-config.exp.guide_depth - depth_inc)])
                    depth_inc += 1
                    n_neighs = len(neigh_labels)
                
                # we get rid of label from leaves as we dont want
                # to guide in its direction
                neigh_labels.pop(neigh_labels.index(label.item()))

                neigh_labels_sample = random.choices(neigh_labels, k = n_inp)
                guidance_classes.append(neigh_labels_sample)

        guidance_classes = torch.tensor(sum(guidance_classes, []))
            
        return guidance_classes