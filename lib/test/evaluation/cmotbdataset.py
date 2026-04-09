import os

import numpy as np

from lib.test.evaluation.data import BaseDataset, Sequence, SequenceList
from lib.test.utils.load_text import load_text


class CMOTBDataset(BaseDataset):
    """CMOTB dataset using the official train/test split files."""

    def __init__(self, split='test'):
        super().__init__()
        self.base_path = self.env_settings.cmotb_path
        self.dataset_root = os.path.join(self.base_path, 'dataset', 'RGBNIR')
        self.sequence_list = self._read_split_file(split)

    def _read_split_file(self, split):
        split_path = os.path.join(self.base_path, '{}.txt'.format(split))
        with open(split_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(sequence_name) for sequence_name in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.dataset_root, sequence_name)
        anno_path = os.path.join(seq_path, 'groundtruth_rect.txt')
        ground_truth_rect = load_text(str(anno_path), delimiter=('\t', ','), dtype=np.float64, backend='numpy')

        frames_path = os.path.join(seq_path, 'img')
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith('.jpg')]
        frame_list.sort()
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'cmotb', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)
