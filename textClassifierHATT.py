import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re
from string import maketrans
from sklearn.model_selection import KFold
import sys
import os
from keras.constraints import maxnorm
import sklearn
from gensim.parsing.preprocessing import STOPWORDS
os.environ['KERAS_BACKEND']='theano'
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import keras
from sklearn.model_selection import KFold
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxoutDense,MaxPooling1D,GaussianNoise, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed,Activation
from keras.models import Sequential ,Model
import sys
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializations
from sklearn.metrics import classification_report , precision_recall_fscore_support,precision_score,recall_score,f1_score
import random
import pdb
from string import punctuation
import math
from my_tokenizer import glove_tokenize
from collections import defaultdict
from data_handler import get_data
from keras.regularizers import l2
from keras import regularizers
from keras import constraints
import pickle
from sklearn.model_selection import train_test_split
#from autocorrect import spell
from keras.optimizers import Adam


reload(sys)
sys.setdefaultencoding('utf8')

word2vec_model = None
freq = defaultdict(int)
vocab, reverse_vocab = {}, {}
EMBEDDING_DIM = 310
train_tweets = {}
test_tweets = {}
MAX_SENT_LENGTH = 0
MAX_SENTS = 0
MAX_NB_WORDS = 20000
VALIDATION_SPLIT = 0.1
INITIALIZE_WEIGHTS_WITH = 'glove'
SCALE_LOSS_FUN = False



class AttLayer(Layer):
    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.init = initializations.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = K.dot(uit, self.u)
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

def batch_gen(X, batch_size):
    n_batches = X.shape[0]/float(batch_size)
    n_batches = int(math.ceil(n_batches))
    end = int(X.shape[0]/float(batch_size)) * batch_size
    n = 0
    for i in xrange(0,n_batches):
        if i < n_batches - 1: 
            batch = X[i*batch_size:(i+1) * batch_size, :]
            yield batch
        
        else:
            batch = X[end: , :]
            n += X[end:, :].shape[0]
            yield batch


def get_embedding(word):
    #return
    try:
        return word2vec_model[word]
    except Exception, e:
        print 'Encoding not found: %s' %(word)
        return np.zeros(EMBEDDING_DIM)

def get_embedding_weights():
    f = open('missed_vocab.txt','w')
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    print len(vocab)
    for k, v in vocab.iteritems():
        try:
            embedding[v] = word2vec_model[k]
        except Exception, e:
            f.write(k+'\n')
            n += 1
            pass
    print "%d embedding missed"%n
    f.close()
    return embedding


def gen_sequence(filename):
    y_map = {
            'joy': 0,
            'anger': 1,
            'surprise': 2,
            'disgust':3,
            'fear':4,
            'sad':5
            }

    X, y = [], []
    flag = True
    if filename == 'tokenized_tweets_train.txt':
        for tweet in train_tweets:
            text = glove_tokenize(tweet['text'].lower())
            seq, _emb = [], []
            for word in text:
                seq.append(vocab.get(word, vocab['UNK']))
            X.append(seq)
            y.append(y_map[tweet['label']])
        return X, y
    else:
        for tweet in test_tweets:
            text = glove_tokenize(tweet['text'].lower())
            seq, _emb = [], []
            for word in text:
                seq.append(vocab.get(word, vocab['UNK']))
            X.append(seq)
            y.append(y_map[tweet['label']])
        return X, y

def select_tweets(filename):
    # selects the tweets as in mean_glove_embedding method
    # Processing
    if filename == 'tokenized_tweets_train.txt':
        train_tweets = get_data('tokenized_tweets_train.txt')
    elif filename == 'tokenized_tweets_test.txt':
        test_tweets = get_data('tokenized_tweets_test.txt')
    tweet_return = []
    if filename == 'tokenized_tweets_train.txt':
        c = 1
        for tweet in train_tweets:
            _emb = 0
            words = glove_tokenize(tweet['text'].lower())
            for w in words:
                if w in word2vec_model:  # Check if embeeding there in GLove model
                    _emb+=1
            c = c+1
            # if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
        print('Tweets selected:', len(tweet_return))
        #pdb.set_trace()
        return tweet_return
    else:
        c = 1
        for tweet in test_tweets:
            _emb = 0
            words = glove_tokenize(tweet['text'].lower())
            for w in words:
                if w in word2vec_model:  # Check if embeeding there in GLove model
                    _emb+=1
            c = c+1
            # if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
        print('Tweets selected:', len(tweet_return))
        #pdb.set_trace()
        return tweet_return



def gen_vocab():
    # Processing
    vocab_index = 1
    for tweet in train_tweets:
        text = glove_tokenize(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        # words = [word for word in words if word not in STOPWORDS]

        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
            freq[word] += 1

    for tweet in test_tweets:
        text = glove_tokenize(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        # words = [word for word in words if word not in STOPWORDS]

        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
            freq[word] += 1
    
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'



def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

def lstm_model(sequence_length, embedding_dim):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model = Sequential()
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=False))
    model.add(Dropout(0.3))#, input_shape=(sequence_length, embedding_dim)))
    model.add(Bidirectional(LSTM(150,return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(150,return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(AttLayer())
    model.add(Dropout(0.5))
    #model.add(MaxoutDense(100, W_constraint=maxnorm(2)))
    #model.add(Dropout(0.5))
    model.add(Dense(6,activity_regularizer=l2(0.0001)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(clipnorm=1, lr=0.001),metrics=['accuracy'])
    print model.summary()
    return model




def train_LSTM(X_train, y_train, X_test,y_test, model,inp_dim, weights,  batch_size=500):
    
    # shuffle_weights(model)
    model.layers[0].set_weights([weights])
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X_train.shape[1]
    y_train = y_train.reshape((len(y_train), 1))
    X_temp = np.hstack((X_train, y_train))
    for epochs in range(25,26):
        for epoch in range(epochs):
            print('Epoch ',epoch,'\n')
            c = 0
            for X_batch in batch_gen(X_temp, 500):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]
                class_weights = None
                try:
                    y_temp = to_categorical(y_temp, nb_classes=6)
                except Exception as e:
                    print e
                    print y_temp
                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
                c = c+1
            y_pred = model.predict(X_test,batch_size=500)
            y_pred = np.argmax(y_pred, axis=1)
            print classification_report(y_test, y_pred)
            print f1_score(y_test, y_pred, average='macro')


np.random.seed(42)
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./ntua_twitter_affect_310.txt')
train_tweets = select_tweets('tokenized_tweets_train.txt')
test_tweets =  select_tweets('tokenized_tweets_test.txt')
gen_vocab()
X_train, y_train = gen_sequence('tokenized_tweets_train.txt')
X_test,y_test = gen_sequence('tokenized_tweets_test.txt')
MAX_SEQUENCE_LENGTH1 = max(map(lambda x:len(x), X_train))
MAX_SEQUENCE_LENGTH2 = max(map(lambda x:len(x), X_test))
MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH1,MAX_SEQUENCE_LENGTH2)
print "max seq length is %d"%(MAX_SEQUENCE_LENGTH)

train_data = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
print train_data.shape
print test_data.shape
y_train = np.array(y_train)
y_test = np.array(y_test)
W = get_embedding_weights()
model = lstm_model(train_data.shape[1], EMBEDDING_DIM)
train_LSTM(train_data, y_train, test_data,y_test,model, EMBEDDING_DIM, W)
