'''
### TODO: look as pad_packed_sequences and pack_pad_sequence to work with LSTM
            IDEAS:
            
            bi-directional LSTM
            deep layered LSTM
            
'''

import torch
import torch.nn
from Dataloader import ReviewDataset, pad_collate, make_data_sampler
from Preprocess import preprocess_data_main
import os
import fasttext
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler #samples from a given list of indices w/o replacement
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from datetime import datetime
from sys import argv

root_path = os.getcwd() + '/'

# device_type = argv[1]

# device = torch.device("cuda:"+device_type if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
#########################   HYPERPARAMS    ############################

# LEARNING_RATE = 5*1e-3
# EPOCHS = 30
# WEIGHT_DECAY = 1e-4
# SAVE_MODEL = False
# BATCH_SIZE = 100
# VALIDATION_SPLIT = .2
# SHUFFLE_DATASET = True
# SEED = 42
# IN_DIM = 300
# HIDDEN_DIM = 300
# NUM_LAYERS = 1
# WORKERS = 40
# BIDIRECTIONAL = True
# FREEZE_EMBEDDING_WEIGHTS = False


#####################################################################

class LSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers, weights_matrix, freeze_weights, bidirectional):
        super(LSTM, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(weights_matrix).float(),
                                                           freeze=freeze_weights)
        self.bidirectional = bidirectional
        embedding_dim = weights_matrix.shape[1]
        self.num_layers = num_layers
        self.hidden_size = hidden_dim
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.D = 2 
        else:
            self.D = 1
        self.fc1 = nn.Linear(hidden_dim*self.D*2, 300) # for [h_0,h_1,c_0,c_1] of size [1,1200]
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        # self.conv = nn.Conv1d(embedding_dim, )
    
    def forward(self, x, x_len, device):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size).to(device)
        x = x.to(device)
        
        # pass through embedding
        x = self.embedding(x)
        
        # Pack input size
        x_packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False).to(device)
        
        output_packed, (h_n,c_n) = self.lstm(x_packed, (h0,c0))
        # out : batch_size x seqlength x hidden_size
        
        # pad sequences 
        out, lens = pad_packed_sequence(output_packed, batch_first=True)
        
        
        # print('GRU output(all timesteps)', out.shape)
        # print(out.shape)
        # print('GRU last timestep output')
        # print(out[-1].shape)
        
        # print(f'Last hidden state = { h_n.shape}')
        # print(f' last hidden state of last time_Step = { h_n[-1].shape}')
        
        # since it is as classification problem, we will grab the last hidden state
        # last_h_timestep = out[:, -1, :]
        # print(h_n[0].shape,c_n[0].shape)
        if self.bidirectional:
            last_h_timestep = torch.concat((h_n[0],h_n[1], c_n[0],c_n[1]), axis=1)
        else:
            last_h_timestep = torch.concat((h_n[0], c_n[0]), axis=1)
        # print(last_h_timestep.shape)
        x = self.fc1(last_h_timestep)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        outp = self.fc3(x)
        return outp


if __name__ == '__main__':
    
    print(f'device is {device}')
    
    # ft = fasttext.load_model('../HW-1/cc.en.300.bin')
    # word_vocab = set(ft.words)
    with open(root_path+'embedding_data/embs_npa_glove_300d.pkl','rb') as f:
        embs = pickle.load(f)
    
    with open(root_path+'embedding_data/glove_300_embedd_matrix_vocab_key_index_mapping.pkl','rb') as f:
        vocab_dict = pickle.load(f)

    dataset = ReviewDataset('train_data.csv', vocab_dict)
    
    train_sampler, valid_sampler = make_data_sampler(dataset, validation_split=VALIDATION_SPLIT,
                                                    shuffle_dataset=SHUFFLE_DATASET, seed=SEED)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=pad_collate,
                             num_workers = WORKERS) 
    validation_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler,
                                   collate_fn=pad_collate, num_workers=WORKERS)
    
    lstm = LSTM(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, weights_matrix=embs,
                freeze_weights=FREEZE_EMBEDDING_WEIGHTS, bidirectional=BIDIRECTIONAL).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # first train model with freezing embedding weights  for 1st EPOCHS
    fit(lstm, train_loader, validation_loader, criterion, optimizer)
    
    # if FREEZE_EMBEDDING_WEIGHTS:
    #     # finetune with making Embeddings trainable True
    #     trained_model = LSTM(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, weights_matrix=embs,
    #             freeze_weights=False).to(device)
    #     trained_model.load_state_dict(torch.load(MODEL_PATH))
    #     criterion = nn.BCEWithLogitsLoss().to(device)
    #     optimizer = torch.optim.Adam(trained_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    #     fit(trained_model, train_loader, validation_loader, criterion, optimizer)

    
    
    # for train_features_pad, train_labels, train_len in train_loader:
    #     # print(f"Feature batch shape: {train_features_pad.size()}")
    #     print(f"Labels batch shape: {torch.Tensor(train_labels).reshape(-1,1).shape}")
    #     print(f"Features Lengths: {train_len}")
    #     x = lstm(train_features_pad, train_len)
    #     print(x, type(x), type(train_labels))
    #     # x_packed = pack_padded_sequence(train_features_pad, train_len, batch_first=True, enforce_sorted=False)
    #     # output_packed, hidden = rnn(x_packed)
    #     # print(x_packed, output_packed, hidden.shape)
    #     # x_unpacked, lens_unpacked = pad_packed_sequence(output_packed, batch_first=True)
    #     # print(x_unpacked, lens_unpacked)
    #     break
    
        
        
        