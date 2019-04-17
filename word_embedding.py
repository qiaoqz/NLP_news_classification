import _pickle as pickle
import os
import time
import sys
import numpy as np
import argparse

def main():
    f = open('glove.6B.100d.txt', 'rb')
    g = open('glove.6B.100d_pickle', 'wb')
    word_dict = {}
    wordvec = []
    for idx, line in enumerate(f.readlines()):
        line = line.decode()
        word_split = line.split(' ')
        word = word_split[0]
        word_dict[word] = idx
        d = word_split[1:]
        d[-1] = d[-1][:-1]
        d = [float(e) for e in d]
        wordvec.append(d)

    embedding = np.array(wordvec)
    pickling = {}
    pickling = {'embedding' : embedding, 'word_dict': word_dict}
    pickle.dump(pickling, g)
    f.close()
    g.close()

def word_id_convert(seq, fname):
    g = open('data_pickle', 'rb')
    pickling = pickle.load(g)
    x_text = pickling[seq]
    seq_y = seq + 3
    y = pickling[seq_y]
    g.close()

    max_document_length = max([len(x.split()) for x in x_text])

    h = open('glove.6B.100d_pickle', 'rb')
    pickling = pickle.load(h)
    word_dict = pickling['word_dict']
    # print （len(word_dict)）
    # sys.exit()
    h.close()
    splitter = [x.split(" ") for x in x_text]
    word_indices = []
    for sentence in splitter:
        word_index = [word_dict[word] if word in word_dict else word_dict['the'] for word in sentence]
        padding = max_document_length -  len(word_index)
        padder = [2 for i in range(padding)]
        word_index = word_index + padder
        word_indices.append(word_index)
        # print word_index
    # print splitter
    word_indices = np.array(word_indices)
    pickle_name = 'word_index_pickle'+'_'+ fname
    w = open(pickle_name, 'wb')
    pickling = {'word_indices': word_indices, 'y': y}
    pickle.dump(pickling, w)
    w.close()



main()
MyList = ['train','valid','test']
for idx, name in enumerate(MyList):
    word_id_convert(idx, name)

