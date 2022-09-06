import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Preprocess import preprocess_data_main
import os
import fasttext
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler #samples from a given list of indices w/o replacement
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn

root_path = os.getcwd() + '/'

BATCH_SIZE = 32
VALIDATION_SPLIT = .2
SHUFFLE_DATASET = True
SEED = 42
EMBEDDING_MODEL_PATH = '../HW-1/cc.en.300.bin'

class ReviewDataset(Dataset):
    def __init__(self, data_path, embedding_vocab_dict):
        self.data = pd.read_csv(root_path + data_path) 
        self.vocab = embedding_vocab_dict
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        text, label = self.data.iloc[idx]
        
        # preprocess text data from Preprocess.py
        text = torch.from_numpy(preprocess_data_main(text, self.vocab))
        
        # label change
        if label == 'positive':
            label = 1
        else:
            label = 0
        
        return text, label

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    # yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy, x_lens

def make_data_sampler(dataset, validation_split=VALIDATION_SPLIT, 
                      shuffle_dataset=SHUFFLE_DATASET, seed=SEED):
    '''
    Splits data into train and validation and make samplers for random sampling over train and test
    dataset
    '''
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # print(len(train_indices), len(val_indices), validation_split)
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


if __name__ == '__main__':
    
    ft = fasttext.load_model(EMBEDDING_MODEL_PATH)
    word_vocab = set(ft.words)
    
    dataset = ReviewDataset('train_data.csv', ft, word_vocab)
    
    train_sampler, valid_sampler = make_data_sampler(dataset)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=pad_collate) 
    # validation_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
    #                                                 sampler=valid_sampler)
    rnn = nn.GRU(300, 10, 1, batch_first=True)
    for train_features_pad, train_labels, train_len in train_loader:
        print(f"Feature batch shape: {train_features_pad.size()}")
        print(f"Labels batch shape: {train_labels}")
        print(f"Features Lengths: {train_len}")
        x_packed = pack_padded_sequence(train_features_pad, train_len, batch_first=True, enforce_sorted=False)
        output_packed, hidden = rnn(x_packed)
        print(x_packed, output_packed, hidden.shape)
        x_unpacked, lens_unpacked = pad_packed_sequence(output_packed, batch_first=True)
        print(x_unpacked, lens_unpacked)
        break



