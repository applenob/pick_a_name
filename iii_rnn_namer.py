# coding=utf-8
"""
@author: cer
加载模型，无则训练
"""

from rnn_theano import *
from ii_data_process import load_training_data, load_sample_training_data
import numpy as np


def generate_name(model, name_len, first_name, X_train, y_train, char_to_index, index_to_char):
    new_name = [char_to_index[first_name]]
    for i in range(name_len-1):
        next_word_probs = model.forward_propagation(new_name)
        samples = np.random.multinomial(1, next_word_probs[-1])
        sampled_word = np.argmax(samples)
        new_name.append(sampled_word)
    name_str = [index_to_char[x] for x in new_name]
    return name_str


if __name__ == '__main__':
    np.random.seed(10)
    # X_train, y_train, char_to_index, index_to_char = load_training_data()
    X_train, y_train, char_to_index, index_to_char = load_sample_training_data()
    char_num = len(char_to_index.keys())
    print char_num

    name_num = 10
    name_len = 3
    first_name = u"宋"
    if first_name not in char_to_index:
        print "暂时不支持这个姓，Sorry！！！"
    else:
        print "支持这个姓，请稍等 ... ..."
        model = RNNTheano(char_num)
        # losses = train_with_sgd(model, X_train, y_train, nepoch=50)
        # save_model_parameters_theano('./data/trained-model-theano.npz', model)
        load_model_parameters_theano('./model/model-80-4776-2017-01-10-21-45-46.npz', model)

        for i in range(name_num):
            name = [first_name]
            name = generate_name(model, name_len, first_name, X_train, y_train, char_to_index, index_to_char)
            print " ".join(name)

