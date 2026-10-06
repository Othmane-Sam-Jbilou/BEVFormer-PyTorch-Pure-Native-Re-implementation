import os
import torch
from torch.utils.data import Dataset


class NuScenesFrameLoader(Dataset):
    def __init__(self, preprocessed_dir='./preprocessed_data', is_test=False, split_idx=379):
        self.preprocessed_dir = preprocessed_dir
        all_files = sorted([f for f in os.listdir(preprocessed_dir) if f.endswith('.pt')])

        if not is_test:
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.preprocessed_dir, self.files[idx])
        return torch.load(file_path, weights_only=False)
