import torch
from Dataloader import ReviewDataset, pad_collate, make_data_sampler
from Preprocess import preprocess_data_main
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pickle
from datetime import datetime
from sys import argv
from LSTM import LSTM
from GRU import GRU
from tqdm import tqdm

root_path = os.getcwd() + '/'

device_type = argv[1]

device = torch.device("cuda:"+device_type if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

#########################   HYPERPARAMS    ############################
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
SEED = 42
IN_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
WORKERS = 40
BIDIRECTIONAL = True
FREEZE_EMBEDDING_WEIGHTS = False


MODEL_PATH = root_path + \
            'saved_models2/20220906_211602_model_30-epochs_2-layers_True-bidir_False-freeze.pth'

print(f'LEARNING_RATE = {LEARNING_RATE}','\n',
     f'BATCH_SIZE = {BATCH_SIZE}','\n',
     f'HIDDEN_DIM = {HIDDEN_DIM}','\n',
     f'NUM_LAYERS = {NUM_LAYERS}','\n',
     f'BIDIRECTIONAL = {BIDIRECTIONAL}','\n',
     f'FREEZE_EMBEDDING_WEIGHTS = {FREEZE_EMBEDDING_WEIGHTS}','\n')
#####################################################################


def compute_acc(y_pred, y_true):
    y_pred = y_pred.to('cpu')
    y_true = y_true.to('cpu')
    
    y_pred = torch.sigmoid(y_pred)
    
    y_pred = (y_pred > 0.5).long()
    acc = ((y_pred == y_true).sum()) / y_pred.shape[0]
    
    return acc

if __name__ == '__main__':
    
    print(f'device is {device}')
    
    # ft = fasttext.load_model('../HW-1/cc.en.300.bin')
    # word_vocab = set(ft.words)
    with open(root_path+'embedding_data/embs_npa_glove_300d.pkl','rb') as f:
        embs = pickle.load(f)
    
    with open(root_path+'embedding_data/glove_300_embedd_matrix_vocab_key_index_mapping.pkl','rb') as f:
        vocab_dict = pickle.load(f)

    dataset = ReviewDataset('test_data_clean.csv',vocab_dict)

    test_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=pad_collate,
                                 num_workers=WORKERS)
    
    model = torch.load(MODEL_PATH).to(device)
    model.eval()
    
    running_acc = 0
    cnt = 0
    
    for idx,data in enumerate(tqdm(test_dataloader)):
        xtest_padded, ytest, xtest_len = data
        ytest = torch.Tensor(ytest).reshape(-1,1)
        ypred = model(xtest_padded, xtest_len, device)
        # print(j)
        batch_acc = compute_acc(ypred, ytest)
        # print(s)
        running_acc += batch_acc
        cnt += 1
    
    print(f'Test Accuracy is {(running_acc / cnt) * 100}%')
            