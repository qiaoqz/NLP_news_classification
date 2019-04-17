import string
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import re
import itertools
from collections import Counter
import _pickle as pickle
import os

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # seperate puctuations from words
    string = string.decode()
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
    # In this case lemmatization is not needed.
    #lmtzr = WordNetLemmatizer()
    #para = " ".join([lmtzr.lemmatize(i) for i in para.split()])
    return para

def extract_clean_words(para):
    """
    Only use the first 300 words as inputs.
    If the news is greater than 200 words but less than 300 words, then we will pad the empty part later.
    If the news is less than 200, we reject the input.
    """
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
        # print("*********************************************")
        # print("length of the news is " +str(len(paras)))
        # paras = " ".join(i for i in paras) 
        # print(paras)
        # print("=======================LOST==================")

def load_data_and_labels_another():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    y_train = []
    y_valid =[]
    y_test = []
    x_train =[]
    x_valid = []
    x_test = []
    index = []
    labels = {}
    topics = ['finance' , 'international', 'legal', 'social','tech','hemp']
    for idx, topic in enumerate(topics):
        folder_name = "data/" + topic
        all_files = os.listdir(folder_name)
        clean_news = []
        # read in files in each topic's folder
        for single_file in all_files:
            raw_data_file_name = os.path.join(folder_name, single_file)
            news = list(open(raw_data_file_name, mode = 'rb').readlines())
            clean_news = clean_news + news
        clean_news = [s.strip() for s in clean_news]
        clean_news = [clean_str(s) for s in clean_news]
        clean_news = [normalizing_str(s) for s in clean_news]
        clean_news = [extract_clean_words(s) for s in clean_news]
        clean_news = [s for s in clean_news if s is not None]
        length_of_news = len(clean_news)
        test_num = int(length_of_news * 0.3)
        valid_num = int(length_of_news * 0.3)
        train_num = length_of_news - test_num - valid_num
        clean_news = np.array(clean_news)

        
        
        #x_text = x_text + clean_news
        if topic == 'finance':
            y_topic = [[1,0,0,0,0,0] for _ in clean_news]
        elif topic == 'international':
            y_topic = [[0,1,0,0,0,0] for _ in clean_news]
        elif topic == 'legal':
            y_topic = [[0,0,1,0,0,0] for _ in clean_news]
        elif topic == 'social':
            y_topic = [[0,0,0,1,0,0] for _ in clean_news]
        elif topic == 'tech':
            y_topic = [[0,0,0,0,1,0] for _ in clean_news]
        elif topic == 'hemp':
            y_topic = [[0,0,0,0,0,1] for _ in clean_news]
        y_topic = np.array(y_topic)

        #randomly shuffle the data and divide them into train, valid and test
        np.random.seed(9)
        indices = np.random.permutation(clean_news.shape[0])
        training_idx = indices[:train_num]
        valid_idx = indices[train_num:train_num+valid_num]
        test_idx = indices[train_num+valid_num: ]
        # Testing: no record is missed.
        #tem = np.concatenate((training_idx, valid_idx, test_idx),axis = 0)
        #print(tem.sort())        
        train_piece_x = list(clean_news[training_idx])
        valid_piece_x = list(clean_news[valid_idx])
        test_piece_x = list(clean_news[test_idx])
        train_piece_y = list(y_topic[training_idx])
        valid_piece_y = list(y_topic[valid_idx])
        test_piece_y = list(y_topic[test_idx])
        y_train = y_train + train_piece_y
        y_valid = y_valid + valid_piece_y
        y_test = y_test + test_piece_y
        x_train = x_train + train_piece_x
        x_valid = x_valid + valid_piece_x
        x_test = x_test + test_piece_x

    # Store the data in data_pickle.
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)
    file = open('data_pickle', 'wb')
    pickle.dump([x_train,x_valid,x_test,y_train,y_valid,y_test], file)
    file.close()
    print("-------------------------------------------------------")
    print("*****Dumped Data_pickle*****")

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

# load_data_and_labels_another()