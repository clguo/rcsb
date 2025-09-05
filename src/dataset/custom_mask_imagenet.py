import glob
import pandas as pd
import json
import sys
from PIL import Image
import numpy as np  
import random

import torch
from torch.utils.data import Dataset

from datasets import load_dataset


import logging
log = logging.getLogger(__name__)

class CustomMaskImagenetDataset(Dataset):
    path_hier_json = 'data/metadata/imagenet/wordnet_hierarchy.json'

    def __init__(
            self,
            split: str,
            path_data: str, 
            path_predictions: str,
            n_samples: int,
            filter_id: int,
    ):
        '''
        """
        Args:
            path_data (str): Path to the folder containing subfolders named after ImageNet IDs.
                             Each subfolder should contain 'image.png' and 'image_mask.png' files.
                             For example:
                             path_data/
                             ├── n01440764/
                             │   ├── image.png
                             │   └── image_mask.png
                             ├── n01443537/
                             │   ├── image.png
                             │   └── image_mask.png
                             └── ...
        """
        '''
        super().__init__()
        assert split in ['train', 'validation']
        self.split = split

        self.img_size = 256
        self.n_samples = n_samples

        self.dataset = load_dataset('imagenet-1k', split = self.split, trust_remote_code = True)
        
        self.data = glob.glob(path_data + '/*')
        self.data = pd.DataFrame(self.data, columns=['path'])
        self.data['idx'] = self.data['path'].apply(lambda x: int(x.split('/')[-1]))
        self.data['image'] = self.data['path'].apply(lambda x: Image.open(x + '/image.png'))
        self.data['mask'] = self.data['path'].apply(lambda x: Image.open(x + '/image_mask.png'))
        self.hier = self.get_hier()

        self.predictions = pd.read_csv(path_predictions, index_col = "idx") if path_predictions else None
        self.length = min(len(self.data), n_samples)
        self.hier = self.get_hier()

        self.filter_data(filter_id)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        index = self.map_index(index)
        item = self.data.loc[index] 
        img, label = item['image'], self.dataset[index]['label']

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

        return torch.from_numpy(img), index, label,  pred_label.item(), self._get_mask_for_index(index)
    
    def _get_mask_for_index(self, index):
        # Get the mask for the given label
        mask = self.data.loc[index, 'mask']
        
        # Resize and crop the mask to match the image processing
        while min(*mask.size) >= 2 * self.img_size:
            mask = mask.resize(
                tuple(x // 2 for x in mask.size), resample=Image.BOX)

        scale = self.img_size / min(*mask.size)
        mask = mask.resize(
            tuple(round(x * scale) for x in mask.size), resample=Image.BICUBIC)

        mask_arr = np.array(mask)
        crop_y = (mask_arr.shape[0] - self.img_size) // 2
        crop_x = (mask_arr.shape[1] - self.img_size) // 2
        mask_arr = mask_arr[crop_y : crop_y + self.img_size, crop_x : crop_x + self.img_size]

        # Convert to binary mask
        binary_mask = (mask_arr > 0).astype(np.float32)
        binary_mask = torch.from_numpy(binary_mask)[None, ...]#.repeat(3, 1, 1)
        log.info(f"Area: {binary_mask.sum() / binary_mask.numel()}")
        return binary_mask

    def filter_data(self, filter_id):

        if filter_id is not None:
            gt_labels = pd.DataFrame(
                self.dataset['label'],
                columns=['gt_label'],
            )

            assert len(gt_labels) == len(self.predictions), \
                "Length of gt_labels and predictions should be same. Probably a wrong predictions file is set."

            if self.predictions is None or (
                self.predictions[gt_labels.gt_label == filter_id]["pred_label"].isna().any()
                ):
                log.error(f'Filtering is not possible without predictions for label: {filter_id}')
                log.error(f'Please compute predictions for gt_label: {filter_id}')
                sys.exit(1)

            joint = self.predictions.join(gt_labels)
            # filter indices where filter_id is the gt label and the prediction is correct
            joint_filtered = joint[(joint.pred_label == filter_id) & (joint.pred_label == joint.gt_label)]
            joint_filtered = joint_filtered[joint_filtered.index.isin(self.data['idx'].values)] 
            filtered_idx = joint_filtered.sort_index().index.values

            # create index mapping
            self.map_index = lambda x: int(filtered_idx[x])
            # change length
            self.length = min(len(filtered_idx), self.n_samples)
            self.data = self.data.set_index('idx', drop=True)

        else:
            # if no filtering, index is mapped with indentity
            self.map_index = lambda x: x

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
    