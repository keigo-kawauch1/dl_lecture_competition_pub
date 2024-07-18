import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning
import warnings
import time


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def apply_lowpass_filter(eeg_data, cutoff=30.0, fs=200):
    filtered_data = np.array([butter_lowpass_filter(channel, cutoff, fs) for channel in eeg_data])
    return filtered_data

def normalize(data):
    return (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

def preprocess_eeg_data(eeg_data, cutoff=30.0, fs=200, pca_components=30, ica_tol=0.05, ica_max_iter=30, ica_random_state=None):
    # ローパスフィルタの適用
    filtered_data = apply_lowpass_filter(eeg_data, cutoff, fs)
    
    # データの正規化
    normalized_data = normalize(filtered_data)

    return normalized_data

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        """
        t_0 = time.time()
        for i in range(len(self.X)):
            if i%100 == 0:
                t_1 = time.time()
                print("----{}/{} processed----estimated_time{}".format(i,len(self.X), (len(self.X)-i)/100 * (t_1-t_0)/60))
                t_0 = t_1
            self.X[i] = self.preprocess(self.X[i])
        """
        
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
    
    def preprocess(self, sample):
        processed_data = torch.tensor(preprocess_eeg_data(sample).reshape(271,281))
        return processed_data


