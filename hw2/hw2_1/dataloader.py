import json
from functools import cache
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision import datasets
import numpy as np

from dictionary import sentence_to_idx
# Maybe use vgg11 (pretrained) for the CNN?
class MLDSVideoDataset(Dataset):
    def __init__(self, labels_file, vid_dir, transform=None, target_transform=None):
        # fix this
        
        with open(labels_file, 'r') as FILE_:
            self.vid_labels = json.load(FILE_)

        self.vid_dir = vid_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.vid_labels)
    
    @cache
    def get_ids(self):
        return [label_id for label_id in self.vid_labels]
    
    def __getitem__(self, idx):
        label = self.vid_labels[idx]
        avi_file = label['id']
        label_caps = label['caption']
        label_caps = sentence_to_idx(label_caps[0])
        #label_caps = [sentence_to_idx(captions) for captions in label_caps]
        label_caps = torch.tensor(label_caps)
        
        #video_feature = torch.load((Path(self.vid_dir)/'feat')/f'{avi_file}.npy')
        video_feature = np.load((Path(self.vid_dir)/'feat')/f'{avi_file}.npy')
        video_feature = torch.from_numpy(video_feature)

        return video_feature, label_caps
    