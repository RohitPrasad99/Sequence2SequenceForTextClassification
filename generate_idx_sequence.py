'''
Ties preprocessing.py and embeddings.py converts every datapoints into a list containing index with glove_vocab_dict
'''
import pickle
import pandas as pd
import numpy as np
from Preprocess import preprocess_data_main
import swifter
import os

root_path = os.getcwd() + '/'

def word2idx(text, word2idx_dict):
    '''
    Takes a sequence of words and converts into sequences of indexes where indexes are from glove_vocab
    dictionary
    '''
    idx_seq = []
    for word in text:
        # adding with 2 as 1st and 2nd elemnt in weight matrix is <pad> and <unk>
        try:
            x = word2idx_dict[word] + 2
        except KeyError:
            x = 1
        idx_seq.append(x)
    idx_seq = np.array(idx_seq)
    return idx_seq

def generate_df_of_idx_from_sequences(dataframe, glove_word2idx_dict):
    '''
    converts each data points from sequence of words to sequences of indexes where indexes are 
    from glove_vocab dictionary
    '''
    
    new_df = dataframe.copy()
    
    #first preprocess each dataframe points
    new_df['review'] = dataframe['review'].swifter.apply(lambda x: preprocess_data_main(x,
                                                         glove_word2idx_dict))
    
    # new_df['review'] = new_df['review'].swifter.apply(lambda x:word2idx(x, glove_word2idx_dict))
    
    new_df['sentiment'] = new_df['sentiment'].swifter.apply(lambda x: 1 if x == 'positive' else 0, axis=1)
    
    return new_df
    
if __name__ == '__main__':
    
    with open('embedding_data/glove_300_embedd_matrix_vocab_key_index_mapping.pkl','rb') as f:
        glove_word2idx_dict = pickle.load(f)
        
    df = pd.read_csv(root_path + 'train_data.csv')
    
    new_df = generate_df_of_idx_from_sequences(df, glove_word2idx_dict)
    
    new_df.to_csv(root_path+'processed_data/processed_train_swifter_py.csv', index=False)