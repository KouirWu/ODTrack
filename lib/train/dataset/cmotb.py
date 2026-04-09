import os
from collections import OrderedDict

import numpy as np
import torch

from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class CMOTB(BaseVideoDataset):
    """CMOTB dataset with train/test splits defined by text files."""

    def __init__(self, root=None, image_loader=jpeg4py_loader, split='train', seq_ids=None, data_fraction=None):
        root = env_settings().cmotb_dir if root is None else root
        super().__init__('CMOTB', root, image_loader)

        self.dataset_root = os.path.join(self.root, 'dataset', 'RGBNIR')
        self.split = split
        self.sequence_list = self._build_sequence_list(split, seq_ids)

        if data_fraction is not None:
            num_keep = max(1, int(len(self.sequence_list) * data_fraction))
            self.sequence_list = self.sequence_list[:num_keep]

    def _build_sequence_list(self, split, seq_ids):
        sequence_list = self._read_split_file(split)
        if seq_ids is None:
            return sequence_list
        return [sequence_list[i] for i in seq_ids]

    def _read_split_file(self, split):
        split_path = os.path.join(self.root, '{}.txt'.format(split))
        with open(split_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def get_name(self):
        return 'cmotb'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.dataset_root, self.sequence_list[seq_id])

    def _read_bb_anno(self, seq_path):
        gt_path = os.path.join(seq_path, 'groundtruth_rect.txt')
        gt = np.loadtxt(gt_path, delimiter='\t', dtype=np.float32)
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        bbox = self._read_bb_anno(self._get_sequence_path(seq_id))
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:06}.jpg'.format(frame_id + 1))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_class_name(self, seq_id):
        return 'generic'

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({
            'object_class_name': 'generic',
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })

        return frame_list, anno_frames, object_meta
