'''
This py script attempts to preprocess multi-line review data.

Steps are:
1) remove HTML tags 
2) renove multiple dots
3) decontract words (eg I'm --> I am, etc)
4) drop proper nouns such as names and lemmatize input words
5) remove punctuations and lower-case 
6) generate embeddings (fasttext makes embeddings at n-gram level for unseen words
                        so not necessary to check with word_vocab)

'''

import pandas as pd
import re
import spacy
import numpy as np
import os
import fasttext

WORD_DIM = 300

root_path = os.getcwd() +'/'

def decontracted(phrase):
    # <br> tags
    phrase = re.sub(r'<.*?>', '', phrase)
    
    # multiple dots
    phrase = re.sub(r'\.+', '.', phrase)
    
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def drop_proper_nouns(phrase):
    nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(phrase)
    
    string = ''
    for tok in doc:
        if tok.pos_ != 'PROPN':
            string += str(tok.lemma_) + ' '
    return string

def remove_punctuations(phrase):
    phrase = re.sub(r'\W+ ',' ',phrase)
    
    phrase = phrase.lower()
    return phrase

def generate_embeddings(phrase, ft, word_vocab, word_dim=WORD_DIM):
    vec = []
    for word in phrase:
        if word in word_vocab:
            vec.append(ft.get_word_vector(word))
    return np.asarray(vec)

def preprocess_data_main(phrase, word_vocab_dict):
    phrase = decontracted(phrase)
    
    # phrase = drop_proper_nouns(phrase)
    
    phrase = remove_punctuations(phrase)
    
    # replace each word with its index from word_vocab
    w2idx = []
    for word in phrase:
        # adding with 2 as 1st and 2nd elemnt in weight matrix is <pad> and <unk>
        try:
            x = word_vocab_dict[word] + 2
        except KeyError:
            x = 1
        w2idx.append(x)
    w2idx = np.array(w2idx)
    # phrase_emb = generate_embeddings(phrase, ft_model, word_vocab)
    
    return w2idx

if __name__ == '__main__':
    data_df = pd.read_csv(root_path + 'train_data.csv')
    
    # replace sentiment label with integer
    data_df['sentiment'] = data_df.sentiment.apply(lambda x: 1 if x == 'positive' else 0)
    
    test_review = data_df.review.iloc[10]
    
#     test_review = decontracted(test_review)
    
#     print('#'*50,'\n\n')
#     print(test_review)
    
#     test_review = drop_proper_nouns(test_review)
    
#     print('#'*50,'\n\n')
#     print(test_review)
    
#     test_review = remove_punctuations(test_review)
    
#     print('#'*50,'\n\n')
#     print(test_review)
    
    ft = fasttext.load_model('../HW-1/cc.en.300.bin')
    word_vocab = set(ft.words)
    
    test_review_emb = preprocess_data_main(test_review, ft, word_vocab)
    
    print('#'*50,'\n\n')
    print(test_review_emb.shape)
    