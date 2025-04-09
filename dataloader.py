from time import time
from utils import set_seed, collate_func, mkdir
from torch.utils.data import DataLoader
import torch
import argparse
import warnings, os
import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
import os
import pickle
from pathlib import Path

class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, dataset):
        self.list_IDs = list_IDs
        self.df = df
        
        base_path = Path('/public_bme2/bme-chengzhen/yangjie/DrugSR/data_embedding') / dataset  
        self.protein_path = base_path / 'protein_embeddings'  
        self.smiles_path = base_path / 'mol_embeddings' 

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        protein_id = self.df.iloc[index]['Protein_id']
        smiles_id = self.df.iloc[index]['SMILES_id']

        protein_embed, protein_mask = self.load_embedding(os.path.join(self.protein_path, f"{protein_id}.pkl"))  
        smiles_embed, smiles_mask = self.load_embedding(os.path.join(self.smiles_path, f"{smiles_id}.pkl"))  

        y = self.df.iloc[index]["Y"]

        return smiles_embed, protein_embed, smiles_mask, protein_mask, y  

    def load_embedding(self, file_path):  
        with open(file_path, 'rb') as f:  
            data = pickle.load(f)  
        return torch.from_numpy(data['embedding']), torch.from_numpy(data['mask']) 

def get_dataloader(data, split, batch_size, num_workers):
    """
    Train the DrugSR model for DTI (Drug-Target Interaction) prediction.

    Parameters:
    ----------
    data : str, optional
        The dataset to use for training. Default is 'biosnap'.
        Choices: ['bindingdb', 'biosnap', 'human']

    split : str, optional
        The splitting strategy for the dataset. Default is 'random'.
        Choices: ['random', 'cold', 'cluster']

    batch_size : int, optional
        The input batch size for training. Default is 128.

    num_workers : int, optional
        The number of worker threads for data loading. Default is 1.

    Returns:
    -------
    None
    """
    dataFolder = f'/public_bme2/bme-chengzhen/yangjie/DrugSR/dataset/{data}'
    dataFolder = os.path.join(dataFolder, str(split))

    if split == 'cluster':
        train_path = os.path.join(dataFolder, 'new_source_train.csv')
        val_path = os.path.join(dataFolder, "new_target_test.csv")
        test_path = os.path.join(dataFolder, "new_target_test.csv")
    else:
        train_path = os.path.join(dataFolder, 'new_train.csv')
        val_path = os.path.join(dataFolder, "new_val.csv")
        test_path = os.path.join(dataFolder, "new_test.csv")
        
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_dataset = DTIDataset(df_train.index.values, df_train, data)
    val_dataset = DTIDataset(df_val.index.values, df_val, data)
    test_dataset = DTIDataset(df_test.index.values, df_test, data)

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_workers,
              'drop_last': True, 'collate_fn':collate_func}
    
    train_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)
    
    return train_generator, val_generator, test_generator


if __name__ == '__main__':
    s = time()
    train_generator, val_generator, test_generator = get_dataloader('biosnap', 'random', batch_size=2, num_workers=5)
    device = 'cuda:0'
    for inputs, masks, labels in train_generator:
        inputs, masks, labels = [item.to(device) for item in inputs], [item.to(device) for item in masks], labels.to(device)
        print(inputs, masks, labels)
        break
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
