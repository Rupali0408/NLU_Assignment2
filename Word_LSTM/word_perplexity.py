from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
 
 
import string
import random
import math
import numpy as np
from numpy import linalg as LA
import matplotlib
 
from sklearn.model_selection import train_test_split
from collections import Counter
import keras
import tensorflow
from keras.utils import * 
from keras.models import *
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pickle import dump
from keras.layers import Dense
from keras.layers import LSTM

from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding



# clean
#tokens = raw_text.split()
#raw_text = ' '.join(tokens)
files = ['austen-emma.txt','austen-persuasion.txt','austen-sense.txt','bible-kjv.txt','blake-poems.txt']
#raw_text = gutenberg.raw(files)

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

raw_text = load_doc('gutenberg_test_5.txt')
# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = keras.preprocessing.text.text_to_word_sequence(doc, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ')
    #tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens

#clean document
tokens = clean_doc(raw_text)
#print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)
print('Total Sequences: %d' % len(sequences))

def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# save sequences to file
out_filename = 'word_sequences_test.txt'
save_doc(sequences, out_filename)

# load the model
model = load_model('word_model.h5')
 
# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
# load
in_filename = 'word_sequences_test.txt'
doc = load_doc(in_filename)
lines = list(doc.split('\n'))
#lines = load_doc('word_sequences_test.txt')



sequences_c = tokenizer.texts_to_sequences(lines)
#sequences = np.zeros((len(sequences_c),51))
sequences = []	
for sequence in  sequences_c:
	if(len(sequence)==51):
		sequences.append(sequence)
		

# separate into input and output
sequences =  np.asarray(sequences)
print(sequences.shape)
X_test, y_test = sequences[:,:-1], sequences[:,-1]
#X =   sequences[:,:-1]
y_test = to_categorical(y_test, num_classes=vocab_size)
 

loss = model.evaluate(X_test,y_test,batch_size=128, verbose=1)
print(loss)


