 
import string
import random
import math
import numpy as np
from numpy import linalg as LA
import matplotlib
from nltk.corpus import gutenberg
#from nltk.book import *
from nltk.tokenize import *
from sklearn.model_selection import train_test_split
from collections import Counter
import keras
import tensorflow
from keras.utils import * 
from keras.models import *
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

from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding

def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
 
# load text
raw_text = load_doc('gutenberg_test_1.txt')
#print(raw_text)
 
# clean
tokens = raw_text.split()
raw_text = ' '.join(tokens)
 
# organize into sequences of characters
length = 50
sequences = list()
for i in range(length, len(raw_text)):
#for i in range(length, 100):
	# select sequence of tokens
	seq = raw_text[i-length:i+1]
	# store
	sequences.append(seq)
print('Total Sequences: %d' % len(sequences))
 
# save sequences to file
out_filename = 'char_sequences_test.txt'
save_doc(sequences, out_filename)


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# load
in_filename = 'char_sequences_test.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')
 
# integer encode sequences of characters
 

# load the model
model = load_model('weights-10.hdf5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))


sequences = list()
for line in lines:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)
 
# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)
 
# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X_test = array(sequences)
y_test = to_categorical(y, num_classes=vocab_size)

loss = model.evaluate(X_test,y_test,batch_size=128, verbose=1)
print(loss)


