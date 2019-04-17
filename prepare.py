import string
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import re
import itertools
from collections import Counter
import _pickle as pickle
import json
import os
import gc



def clean_str(string):
    """
    string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub("(www\S*)\s", " ", string)
    string = re.sub(r"[^A-Za-z(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r'"', ' ', string)
    string = string.replace("\'ve", "have").replace("\'d","had").replace("\'s","is")
    string = string.replace("n\'t","not").replace("\'re","are").replace("\'ll","will")
    string = string.replace("\n"," ").replace("\("," ").replace("\)"," ").replace("\?"," ")
    return string.strip().lower()



def normalizing_str(para):
    # Eliminate punctuation
    translator = str.maketrans('', '', string.punctuation)
    para = para.translate(translator)
    # Lemmatization
    #lmtzr = WordNetLemmatizer()
    #para = " ".join([lmtzr.lemmatize(i) for i in para.split()])
    return para

def extract_clean_words(para):
    paras = para.split()
    noise = ["http","apnews","news","link","subscribe","dc","d","c","s"]
    for x in noise:
        if x in para.split():
           paras.remove(x)
    

    if (len(paras) <= 300) & (len(paras) >= 200):
        words_hundred = ""
        for i in paras:
            words_hundred = words_hundred + " " + i

        return words_hundred.strip()
    elif len(paras) > 300:
        words_hundred = ""
        for i in range(300):
            words_hundred = words_hundred + " " + paras[i]
        return words_hundred.strip()
    else:
        pass
        # print("========================================")
        # print("length of the news is " +str(len(paras)))
        # paras = " ".join(i for i in paras) 
        # print(paras)
        # print("Unqualified Content, Web Scraping Failed")


def load_data_and_labels_another(rawdata):
    """
    Clean the data
    """

    clean_news = rawdata.strip()
    clean_news = clean_str(clean_news)
    clean_news = normalizing_str(clean_news)
    clean_news = extract_clean_words(clean_news)

    # print("***** Cleaned Data *****")
    return clean_news

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def word_id_convert(x_text):
    """
    Covert the words of our data into numbers according to the pretrain data of GloVe.
    Please keep glove.6B.100d_pickle in the same folder with scripts.
    """
    max_document_length = 300

    h = open('glove.6B.100d_pickle', 'rb')
    pickling = pickle.load(h)
    word_dict = pickling['word_dict']
    h.close()
    splitter = x_text.split(" ")
    
    word_index = [word_dict[word] if word in word_dict else word_dict['the'] for word in splitter]
    padding = max_document_length -  len(word_index)
    padder = [2 for i in range(padding)]
    word_index = word_index + padder



    word_index = np.array(word_index)
    return word_index

