import torch
from Dataloader import ReviewDataset, pad_collate, make_data_sampler
from Preprocess import preprocess_data_main
import os
import fasttext
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler #samples from a given list of indices w/o replacement
import torch.nn as nn
import pickle
from datetime import datetime
from sys import argv
from LSTM import LSTM
from GRU import GRU

root_path = os.getcwd() + '/'

device_type = argv[1]

device = torch.device("cuda:"+device_type if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

#########################   HYPERPARAMS    ############################
MODEL = argv[2]
LEARNING_RATE = 1e-3
EPOCHS = 30
WEIGHT_DECAY = 1e-4
SAVE_MODEL = False
BATCH_SIZE = 32
VALIDATION_SPLIT = .2
SHUFFLE_DATASET = True
SEED = 42
IN_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
WORKERS = 40
BIDIRECTIONAL = True
FREEZE_EMBEDDING_WEIGHTS = False
PER_BATCH_PRINT = 200 #print av. training loss and av. acc after PER_BATCH_PRINT number of batches


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
MODEL_PATH = root_path + \
            'saved_models2/{}_model_{}-epochs_{}-layers_{}-bidir_{}-freeze.pth'.format(timestamp,
                                           EPOCHS,NUM_LAYERS, BIDIRECTIONAL,FREEZE_EMBEDDING_WEIGHTS)

print(f'MODEL = {MODEL}','\n',
     f'LEARNING_RATE = {LEARNING_RATE}','\n',
     f'BATCH_SIZE = {BATCH_SIZE}','\n',
     f'HIDDEN_DIM = {HIDDEN_DIM}','\n',
     f'NUM_LAYERS = {NUM_LAYERS}','\n',
     f'EPOCHS = {EPOCHS}','\n',
     f'WEIGHT_DECAY = {WEIGHT_DECAY}','\n',
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

def train_one_epoch(model, train_loader, criterion, optimizer):
    '''
    In training model for one epoch loop 
    '''
    running_loss = 0.
    last_loss = 0.
    running_acc = 0.
    total_loss = []
    total_acc = []
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        padded_inputs, labels, unpadded_input_len = data

        labels = torch.Tensor(labels).reshape(-1,1).to(device)


        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(padded_inputs, unpadded_input_len, device)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        total_loss.append(loss.item())
        
        # Training accuracy
        acc = compute_acc(outputs, labels)
        running_acc += acc
        total_acc.append(acc)
        
        
        if i % PER_BATCH_PRINT == (PER_BATCH_PRINT-1) or i == 0: 
            last_loss = running_loss / ((i % PER_BATCH_PRINT) + 1) # loss per batch
            last_acc = running_acc / ((i % PER_BATCH_PRINT) + 1) # acc per batch
            print('batch {} loss: {} accuracy: {}'.format(i + 1, last_loss, last_acc))
            running_loss = 0.
            running_acc = 0.
        
    # compute average of loss and accuracy in 1 epoch
    total_loss = np.mean(np.asarray(total_loss))
    total_acc = np.mean(np.asarray(total_acc))
        
    return total_loss, total_acc


def fit(model, train_loader, validation_loader, criterion, optimizer, epochs=EPOCHS, model_path=MODEL_PATH):
    '''
    Since validation loader and train loader can't be accessed simulatneously
    In 1 train epoch loop, we use validatio loader loop to view validation error per epoch
    '''
    best_vloss = 1e9
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, avg_acc = train_one_epoch(model, train_loader, criterion, optimizer)

        # We don't need gradients on to do reporting
        model.eval()

        running_vloss = 0.0
        running_vacc = 0.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vpadded_inputs, vlabels, v_unpadded_input_len = vdata
                vlabels = torch.Tensor(vlabels).reshape(-1,1).to(device)

                voutputs = model(vpadded_inputs, v_unpadded_input_len, device)
                vloss = criterion(voutputs, vlabels)
                vacc = compute_acc(voutputs, vlabels)
                running_vloss += vloss
                running_vacc += vacc
                
            avg_vloss = running_vloss / (i + 1)
            avg_vacc = running_vacc / (i + 1)

            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            print('ACCURACY train {} valid {}'.format(avg_acc, avg_vacc))

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(model, model_path)
                print(f'Model epoch ={epoch+1} saved at {model_path}!!!','#'*100)    

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
    
    if MODEL == 'lstm':
        model = LSTM(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, weights_matrix=embs,
                    freeze_weights=FREEZE_EMBEDDING_WEIGHTS, bidirectional=BIDIRECTIONAL).to(device)
    else:
         model = GRU(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, weights_matrix=embs,
                    freeze_weights=FREEZE_EMBEDDING_WEIGHTS, bidirectional=BIDIRECTIONAL).to(device)
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # first train model with freezing embedding weights  for 1st EPOCHS
    fit(model, train_loader, validation_loader, criterion, optimizer)
    
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
