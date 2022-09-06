'''
This file generates a embedding matrix with glove embeddings so that it can be used in nn.embedding layer

'''
import pickle
import numpy as np
import os
import torch.nn

root_path = os.getcwd() + '/'



if __name__ == "__main__":
    with open(root_path+'embedding_data/glove_300_embedd_matrix.pkl','rb') as f:
        embs = pickle.load(f)
    
    with open(root_path+'embedding_data/glove_300_embedd_matrix_vocab_key_index_mapping.pkl','rb') as f:
        vocab_dict = pickle.load(f)
        
    vocab = np.array(list(vocab_dict.keys()))
    
    #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    vocab = np.insert(vocab, 0, '<pad>')
    vocab = np.insert(vocab, 1, '<unk>')
    print(vocab[:10])

    pad_emb_npa = np.zeros((1,embs.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs,axis=0,keepdims=True)    #embedding for '<unk>' token.

    #insert embeddings for pad and unk tokens at top of embs_npa.
    embs = np.vstack((pad_emb_npa,unk_emb_npa,embs))
    print(embs.shape)
    
    with open(root_path+'embedding_data/vocab_npa_glove_300d.pkl','wb') as f:
        pickle.dump(vocab,f)

    with open(root_path+'embedding_data/embs_npa_glove_300d.pkl','wb') as f:
        pickle.dump(embs,f)
        
    my_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs).float())

    assert my_embedding_layer.weight.shape == embs.shape
    print(my_embedding_layer.weight.shape)
    