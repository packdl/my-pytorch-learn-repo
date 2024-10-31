import json
from functools import cache
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np

from dictionary import BertDictionary


class BertDataset(Dataset):
    def __init__(self, dataset_file, transform = None, target_transform=None):
        with open(dataset_file,'r') as FILE_:
            self.dataset_file = dataset_file
            self.dictionary = BertDictionary(dataset_file)
            self.transform = transform
            self.target_transform = target_transform


    @cache
    def __len__(self):
        return self.dictionary.num_qapairs()                
    
    def __getitem__(self, idx):
        groups = self.dictionary.get_data_groups()
        print(len(groups))
        return groups[idx]

        
if __name__=='__main__':
    bds = BertDataset('data/spoken_test-v1.1.json')
    print(len(bds))

    bds2 = BertDataset('data/spoken_train-v1.1.json')
    print(len(bds2))

    print(bds2[2])
    print(bds[2])
    bds2[1]
    bds2[5]
    bds[5]


""" class MLDSVideoDataset(Dataset):
    def __init__(self, labels_file, vid_dir, transform=None, target_transform=None):
        # fix this
        
        with open(labels_file, 'r') as FILE_:
            self.vid_labels = json.load(FILE_)

        self.vid_dir = vid_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        for paragraph in s
        return len(self.vid_labels['data'])
    
    @cache
    def get_ids(self):
        return [label_id for label_id in self.vid_labels]
    
    def __getitem__(self, idx):
        label = self.vid_labels[idx]
        avi_file = label['id']
        label_caps = label['caption']
        label_caps = sentence_to_idx(label_caps[0])
        label_caps = torch.tensor(label_caps)
        #video_feature = torch.load((Path(self.vid_dir)/'feat')/f'{avi_file}.npy')
        video_feature = np.load((Path(self.vid_dir)/'feat')/f'{avi_file}.npy')
        video_feature = torch.from_numpy(video_feature)

        return video_feature, label_caps

class WeirdDataset(Dataset):
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
            label_caps = torch.tensor(label_caps)
            #video_feature = torch.load((Path(self.vid_dir)/'feat')/f'{avi_file}.npy')
            video_feature = np.load((Path(self.vid_dir)/'feat')/f'{avi_file}.npy')
            video_feature = torch.from_numpy(video_feature)

            return (avi_file, video_feature), label_caps

     """