# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:06:59 2021

@author: clovi
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPool1D, LSTM, Bidirectional, Embedding, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score


MAX_SEQUENCE_LENGHT = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

print('Loading word vectors....')
word2vec = {}
with open(os.path.join('D:/clovi/Estudos/Deep-Learning-Projects/CNN/Toxic/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM), encoding = 'utf8') as f:
    #  it is just a space-separated text file in the format:
    #  word vec[0] vec[1] vec[2] ...
    for line in f:
        values = line.split()
        word = values[0]
        vec=np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))


# Prepare text samples and labels
print('Loading in comments...')

train = pd.read_csv('D:/clovi/Estudos/Deep-Learning-Projects/CNN/Toxic/train.csv')
sentences = train['comment_text'].fillna('DUMMY_VALUE').values
possible_labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
targets = train[possible_labels].values

print('max sequence length:', max(len(s) for s in sentences))
print('min sequence length:', min(len(s) for s in sentences))
s = sorted(len(s) for s in sentences)
print('median sequence length:', s[len(s)//2])

# convert the sentenes (strings) into integers
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
# print('sequences:', sequences); exit()

# get word -> integer mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))

# pad sequences so that we get a N x T matrix
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGHT)
print('Shape of data tensor:', data.shape)

#  prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all zeros
            embedding_matrix[i] = embedding_vector
            
# load pre-trained word embeddings into an Embedding Layer
# note that we set trainable = False so as to keep embeddings fixed
embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGHT,
    trainable=False
    )

print('Building model...')

# create an LSTM network with a single LSTM
input_ = Input(shape = (MAX_SEQUENCE_LENGHT,))
x = embedding_layer(input_)
x = LSTM(15, return_sequences=True)(x)
# x = Bidirectional(LSTM(15, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
output = Dense(len(possible_labels),activation = 'sigmoid')(x)

model = Model(input_, output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Training model...')
r = model.fit(data, targets, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = VALIDATION_SPLIT)
 
# plotting some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['accuracy'], label='val_acc')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show() 

# plot the mean AUC over each label
p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))