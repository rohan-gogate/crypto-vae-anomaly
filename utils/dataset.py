import torch 
from torch.utils.data import Dataset
print("Loading dataset...")
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        self.data = torch.tensor(sequences, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index_):
        return self.data[index_], self.data[index_]
    