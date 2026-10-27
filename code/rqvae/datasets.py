import numpy as np
import torch
import json
import torch.utils.data as data

# This dataset is used to load the embeddings of query_sentences
class EmbDataset(data.Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.dim = self.embeddings.shape[-1]
        self.shape = self.embeddings.shape

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)

    def __repr__(self):
        return f"EmbDataset(path={self.data_path}, shape={self.shape})"
    
class PairedDataset(data.Dataset):
    def __init__(self, ds_a, ds_b):
        assert len(ds_a) == len(ds_b), "两个 Dataset 长度必须相同"
        self.ds_a = ds_a
        self.ds_b = ds_b
    def __len__(self):
        return len(self.ds_a)
    def __getitem__(self, idx):
        return self.ds_a[idx], self.ds_b[idx]


# This dataset is used to initialize the embeddings for the RQVAE model.
class Dataset_init(data.Dataset):
    def __init__(self,data_path):

        self.data_path = data_path
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb, index

    def __len__(self):
        return len(self.embeddings)
    
class TestDataset(data.Dataset):
    def __init__(self, data_path):
        values_list = np.load(data_path)
        self.data = torch.stack([torch.tensor(v) for v in values_list], dim=0)
        # self.data = self.data[:, 1:, :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

