from nltk.tokenize import TweetTokenizer
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
from keras.layers import Embedding,InputLayer
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxoutDense,MaxPooling1D,GaussianNoise, GlobalMaxPooling1D,Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed,Activation
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
from my_tokenizer import glove_tokenize,word_frame_tokenize
from collections import defaultdict
from data_handler import get_data
from keras.regularizers import l2
from keras import regularizers
from keras import constraints
import pickle
from sklearn.model_selection import train_test_split
#from autocorrect import spell
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


reload(sys)
sys.setdefaultencoding('utf8')

word2vec_model = None
freq = defaultdict(int)
vocab, reverse_vocab = {}, {}
frame_vocab ,reverse_frame_vocab = {},{}
EMBEDDING_DIM = 310
train_tweets = {}
test_tweets = {}
MAX_SENT_LENGTH = 0
MAX_SENTS = 0
MAX_NB_WORDS = 20000
VALIDATION_SPLIT = 0.1
INITIALIZE_WEIGHTS_WITH = 'glove'
SCALE_LOSS_FUN = False
liwc_features_num = 39



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

def get_frame_embedding_weights():
    embedding = np.zeros((len(frame_vocab) + 1, 20))
    n = 0
    for k, v in frame_vocab.iteritems():
        try:
            embedding[v] = np.random.uniform(low=-0.05, high=0.05, size=20)
        except Exception, e:
            #f.write(k+'\n')
            n += 1
            pass
    print "%d embedding missed"%n
    #f.close()
    return embedding


def gen_tweet_frame_sequence(filename):
    y_map = {
            'joy': 0,
            'anger': 1,
            'surprise': 2,
            'disgust':3,
            'fear':4,
            'sad':5
            }

    X, X_fr,y, = [], [],[]
    flag = True
    if filename == 'tokenized_tweets_train.txt':
        for tweet, frame in train_tweets:
            text,frame = word_frame_tokenize(tweet['text'].lower(),frame)
            seq, fra = [], []
            for word in text:
                seq.append(vocab.get(word, vocab['UNK']))
            for word in frame:
                fra.append(frame_vocab.get(word, frame_vocab['UNK']))
            X.append(seq)
            X_fr.append(fra)
            y.append(y_map[tweet['label']])
        return X,X_fr, y
    else:
        for tweet, frame in test_tweets:
            text,frame = word_frame_tokenize(tweet['text'].lower(),frame)
            seq, fra = [], []
            for word in text:
                seq.append(vocab.get(word, vocab['UNK']))
            for word in frame:
                fra.append(frame_vocab.get(word, frame_vocab['UNK']))
            X.append(seq)
            X_fr.append(fra)
            y.append(y_map[tweet['label']])
        return X,X_fr, y


def select_tweet_frame(filename):
    if filename == 'tokenized_tweets_train.txt':
        train_tweets = get_data('tokenized_tweets_train.txt')
    elif filename == 'tokenized_tweets_test.txt':
        test_tweets = get_data('tokenized_tweets_test.txt')
    tweet_return = []
    if filename == 'tokenized_tweets_train.txt':
        for tweet, frame in zip(train_tweets,open('frames.txt','r')):
            tweet_return.append((tweet,frame.strip()))
        print('Tweets selected:', len(tweet_return))
        return tweet_return
    else:
        for tweet, frame in zip(test_tweets,open('frames_test.txt','r')):
            tweet_return.append((tweet,frame.strip()))
        print('Tweets selected:', len(tweet_return))
        return tweet_return



def gen_vocab():
    # Processing
    vocab_index = 1
    for tweet,frame in train_tweets:
        text,frame = word_frame_tokenize(tweet['text'].lower(),frame)
        #text = ' '.join([c for c in text if c not in punctuation])
        words = text
        # words = [word for word in words if word not in STOPWORDS]

        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
            freq[word] += 1

    for tweet,frame in test_tweets:
        text,frame = word_frame_tokenize(tweet['text'].lower(),frame)
        #text = ' '.join([c for c in text if c not in punctuation])
        words = text
        # words = [word for word in words if word not in STOPWORDS]

        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
            freq[word] += 1
    
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'



def gen_frame_vocab():
    vocab_index = 1
    for frames in open('frames.txt','r'):
        frames = frames.strip().split()
        for frame in frames:
            if frame not in frame_vocab:
                frame_vocab[frame] = vocab_index
                reverse_frame_vocab[vocab_index] = frame       # generate reverse vocab as well
                vocab_index += 1
            # freq[word] += 1

    for frames in open('frames_test.txt','r'):
        frames = frames.strip().split()
        for frame in frames:
            if frame not in frame_vocab:
                frame_vocab[frame] = vocab_index
                reverse_frame_vocab[vocab_index] = frame       # generate reverse vocab as well
                vocab_index += 1
    
    frame_vocab['UNK'] = len(frame_vocab) + 1
    reverse_frame_vocab[len(frame_vocab)] = 'UNK'
    print('Len of frame vocab',len(frame_vocab))



def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

def lstm_model(sequence_length, word_embedding_matrix, frame_embedding_matrix,embedding_dim):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model1 = Sequential()
    model1.add(Embedding(len(vocab)+1, embedding_dim,weights= [word_embedding_matrix], input_length=sequence_length, trainable=False))
    #model1.add(Flatten())

    model2 = Sequential()
    model2.add(Embedding(len(frame_vocab)+1, 20,weights= [frame_embedding_matrix], input_length=sequence_length, trainable=True))
    #model2.add(Flatten())

    model3 = Sequential()
    model3.add(Merge([model1, model2], mode='concat'))
    model3.add(Dropout(0.3))
    model3.add(Bidirectional(LSTM(150,return_sequences=True)))
    model3.add(Dropout(0.3))
    model3.add(Bidirectional(LSTM(150,return_sequences=True)))
    model3.add(Dropout(0.3))
    #model1.add(Bidirectional(LSTM(150,return_sequences=True)))
    #model1.add(Dropout(0.3))
    #model1.add(Flatten())
    model3.add(AttLayer())

    #model3.add(Flatten())
    model3.add(MaxoutDense(100, W_constraint=maxnorm(2)))
    model3.add(Dropout(0.5))
    model3.add(Dense(6,activity_regularizer=l2(0.0001)))
    model3.add(Activation('softmax'))
    model3.compile(loss='categorical_crossentropy',  optimizer=adam, metrics=['accuracy'])
    print(model3.summary())
    
    return model3


def train_LSTM_with_frame(X_train, y_train, X_test,y_test,X_frame_train, X_frame_test,model):
    best_macro = 0.0
    y_train = to_categorical(y_train,nb_classes=6)
    checkpointer = ModelCheckpoint(filepath='./weights1.hdf5', verbose=1, save_best_only=True)
    model.fit(x=[X_train,X_frame_train],y=y_train,batch_size=500, nb_epoch=30, validation_data=([X_test, X_frame_test], to_categorical(y_test,nb_classes=6)),callbacks=[checkpointer])
    model.load_weights(weightsPath)
    y_pred = model.predict([X_test,X_frame_test],batch_size=500)
    y_pred = np.argmax(y_pred, axis=1)
    print classification_report(y_test, y_pred)
    if f1_score(y_test, y_pred, average='macro')>=best_macro:
        f = open('diff_predictions1.txt','w')
        m = {0:'joy',1:'anger',2:'surprise',3:'disgust',4:'fear',5:'sad'}
        c = 0
        for p in range(len(y_pred)):
            f.write(m[y_pred[p]]+'\t'+str(c+1)+'\n')
            c = c+1
        f.close()
        best_macro = f1_score(y_test,y_pred,average='macro')
        print('Best macro so far = ',best_macro)
    if epoch%10==0 and epoch>0:
        K.set_value(adam.lr, 0.5 * K.get_value(adam.lr))
    print(epoch, K.get_value(adam.lr))


np.random.seed(42)
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('ntua_twitter_affect_310.txt')
train_tweets = select_tweet_frame('tokenized_tweets_train.txt')
test_tweets =  select_tweet_frame('tokenized_tweets_test.txt')
gen_vocab()
gen_frame_vocab()

X_train,X_train_frame, y_train = gen_tweet_frame_sequence('tokenized_tweets_train.txt')
X_test,X_test_frame,y_test = gen_tweet_frame_sequence('tokenized_tweets_test.txt')


MAX_SEQUENCE_LENGTH1 = max(map(lambda x:len(x), X_train))
MAX_SEQUENCE_LENGTH2 = max(map(lambda x:len(x), X_test))
MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH1,MAX_SEQUENCE_LENGTH2)
print "max seq length is %d"%(MAX_SEQUENCE_LENGTH)

train_data = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

train_frame_data = pad_sequences(X_train_frame, maxlen=MAX_SEQUENCE_LENGTH)
test_frame_data = pad_sequences(X_test_frame, maxlen=MAX_SEQUENCE_LENGTH)


print train_data.shape
print test_data.shape



y_train = np.array(y_train)
y_test = np.array(y_test)


W = get_embedding_weights()
W_fr = get_frame_embedding_weights()


adam = Adam(clipnorm=1,lr =.001)
print(train_data[0])
print(train_frame_data[0])
model = lstm_model(train_data.shape[1], W,W_fr,EMBEDDING_DIM)
train_LSTM_with_frame(train_data, y_train, test_data,y_test,train_frame_data,test_frame_data,model)
