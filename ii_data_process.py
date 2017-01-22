# coding=utf-8
"""
@ author: cer
数据预处理
"""
import numpy as np
import os
import cPickle as pickle
import random


def load_training_data():
    """从pkl加载训练数据，无则通过语料生成"""
    unknown_token = "UNKNOWN_TOKEN"
    name_start_token = "NAME_START"
    name_end_token = "NAME_END"

    # 尝试加载pkl
    train_pkl_name = "training_data/train.pkl"
    if os.path.exists(train_pkl_name):
        with open(train_pkl_name, 'rb') as f:
            return pickle.load(f)

    # 读取原始语料文件
    print "Reading PKL Files ..."
    raw_pkl_name = "data/detail_data.pkl"
    with open(raw_pkl_name, 'rb') as f:
        data = pickle.load(f)
    names = []
    chars = set()
    for datum in data:
        names += [[name_start_token] + list(x) + [name_end_token] for x in datum["names"]]
        for name in datum["names"]:
            for one in name:
                chars.add(one)
    print "Parsed %d names." % (len(names))
    # generate char_to_index and index_to_char
    char_to_index = {name_start_token: 0, name_end_token: 1}
    i = 2
    for char in chars:
        char_to_index[char] = i
        i += 1
    index_to_char = dict([(char_to_index[c], c) for c in char_to_index])

    # Create the training data
    X_train = np.asarray([[char_to_index[c] for c in name[:-1]] for name in names])
    y_train = np.asarray([[char_to_index[c] for c in name[1:]] for name in names])

    # 保存到pkl
    all = [X_train, y_train, char_to_index, index_to_char]
    with open(train_pkl_name, "wb") as f:
        pickle.dump(all, f)
    return all


def load_sample_training_data(type):
    """(缩小数据集)从pkl加载训练数据，无则通过语料生成"""
    SAMPLE_RATE = 0.2
    LEAST_NUM = 250

    # 尝试加载pkl
    train_pkl_name = "training_data/"+str(type)+"_train_sample.pkl"
    if os.path.exists(train_pkl_name):
        with open(train_pkl_name, 'rb') as f:
            return pickle.load(f)

    # 读取原始语料文件
    print "Reading PKL Files ..."
    raw_pkl_name = "data/"+str(type)+"_detail_data.pkl"
    with open(raw_pkl_name, 'rb') as f:
        data = pickle.load(f)
    names = []
    chars = set()
    for datum in data:
        sample_num = min(datum["num"], max(int(datum["num"]*SAMPLE_RATE), LEAST_NUM))
        # print sample_num
        sample_names = random.sample(datum["names"], sample_num)
        for name in sample_names:
            names.append(name)
            for one in name:
                chars.add(one)
    print "Parsed %d names." % (len(names))
    # generate char_to_index and index_to_char
    char_to_index = {}
    i = 1
    for char in chars:
        char_to_index[char] = i
        i += 1
    index_to_char = dict([(char_to_index[c], c) for c in char_to_index])

    # Create the training data
    X_train = np.asarray([[char_to_index[c] for c in name[:-1]] for name in names])
    y_train = np.asarray([[char_to_index[c] for c in name[1:]] for name in names])

    # 保存到pkl
    all = [X_train, y_train, char_to_index, index_to_char]
    with open(train_pkl_name, "wb") as f:
        pickle.dump(all, f)
    return all


def load_bin_vec(fname):
    """
    加载 400x1 自训练的char2vecs。
    char2vecs是一个dict，key是word，value是vector。
    """
    char2vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        char_num, vector_dim = map(int, header.split())
        print char_num, vector_dim
        # binary_len是char2vec的字节数
        binary_len = np.dtype('float32').itemsize * vector_dim
        for line in xrange(char_num):
            ch = f.read(3)
            char2vecs[ch] = np.fromstring(f.read(binary_len), dtype='float32')
    return char2vecs


def get_W(char_to_index, index_to_char, k=400):
    """
    接收c2v，相当于把c2v从字典转换成矩阵W。
    相当于原来从char到vector只用查阅c2v字典；
    现在需要先从char_to_index查阅word的索引，再用char的索引到W矩阵获取vector。
    """
    char2vecs = load_bin_vec("vector/wiki.cn.text.jian.vector")
    print char2vecs
    vocab_size = len(char2vecs)
    print vocab_size
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for char in char_to_index:
        if char in char2vecs:
            W[i] = char2vecs[char][:k]
        else:
            W[i] = np.random.uniform(-1, 1, (k,))
        i += 1
    return W


def print_train_example():
    """查看部分训练数据"""
    X_train, y_train, char_to_index, index_to_char = load_sample_training_data(0)
    x_example, y_example = X_train[17], y_train[17]
    print
    print "x:\n%s\n%s" % (" ".join([index_to_char[x] for x in x_example]), x_example)
    print "\ny:\n%s\n%s" % (" ".join([index_to_char[y] for y in y_example]), y_example)

if __name__ == '__main__':
    # print_train_example()
    X_train, y_train, char_to_index, index_to_char = load_sample_training_data(0)
    get_W(char_to_index, index_to_char, k=300)
