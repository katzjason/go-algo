import torch
import os
from torch.utils.data import Dataset, DataLoader

class GoChunkedDataset(Dataset):
    def __init__(self, chunk_folder):
        self.cached_chunk = None
        self.cached_chunk_idx = None
        self.chunk_paths = [
            os.path.join(chunk_folder, f)
            for f in os.listdir(chunk_folder)
            if f.endswith(".pt")
        ]
        
        self.samples = []  # (file_idx, local_idx) mapping

        # Load file sizes
        self.chunk_metadata = []
        for i, path in enumerate(self.chunk_paths):
            data = torch.load(path)
            size = len(data['X'])
            self.chunk_metadata.append((i, size))
            for j in range(size):
                self.samples.append((i, j))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, local_idx = self.samples[idx]

        if self.cached_chunk_idx != file_idx:
            self.cached_chunk = torch.load(self.chunk_paths[file_idx])
            self.cached_chunk_idx = file_idx

        x = self.cached_chunk['X'][local_idx] # [10, 9, 9]
        y_policy = torch.tensor(self.cached_chunk['y_policy'][local_idx], dtype=torch.long) # int (0–80)
        y_value = self.cached_chunk['y_value'][local_idx]
        if y_value == -3.0:
            y_value = -1.0
        if idx < 5:
            print(f"y_value: {y_value}")
        y_value = torch.tensor(y_value, dtype=torch.float) # float (-1 to 1)
        
        return x, y_policy, y_value
    
