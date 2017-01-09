# coding=utf-8
"""
@author: cer
加载模型，无则训练
"""

from rnn_theano import *
from ii_data_process import load_training_data
import numpy as np


def generate_sentence(model, X_train, y_train, char_to_index, index_to_char):
    unknown_token = "UNKNOWN_TOKEN"
    name_start_token = "NAME_START"
    name_end_token = "NAME_END"
    # We start the sentence with the start token
    new_sentence = [char_to_index[name_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == char_to_index[name_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = char_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == char_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [char_to_index[x] for x in new_sentence[1:-1]]
    return sentence_str



if __name__ == '__main__':
    np.random.seed(10)
    X_train, y_train, char_to_index, index_to_char = load_training_data()
    char_num = len(char_to_index.keys())
    model = RNNTheano(char_num)
    # losses = train_with_sgd(model, X_train, y_train, nepoch=50)
    # save_model_parameters_theano('./data/trained-model-theano.npz', model)
    load_model_parameters_theano('./model/model-80-6633-2017-01-09-22-59-47.npz', model)

    num_sentences = 10
    senten_min_length = 7

    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = generate_sentence(model, X_train, y_train, char_to_index, index_to_char)
        print " ".join(sent)

