#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
from ii_data_process import load_training_data, load_sample_training_data

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '20'))
_MODEL_FILE = os.environ.get('MODEL_FILE')


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        print "[ {} ] training epoch no.{} ...".format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), epoch)
        if epoch % evaluate_loss_after == 0:
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("./model/model-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

if __name__ == '__main__':
    # X_train, y_train, char_to_index, index_to_char = load_sample_training_data()
    X_train, y_train, char_to_index, index_to_char = load_training_data()
    char_num = len(char_to_index.keys())

    vocabulary_size = char_num
    model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
    t1 = time.time()
    model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
    t2 = time.time()
    print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

    if _MODEL_FILE != None:
        load_model_parameters_theano(_MODEL_FILE, model)

    train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)
