# coding=utf-8
"""
LSTM (预训练c2v) 实现，以及模型训练，使用Tensorflow
"""

import os
import numpy as np
import random
import string
import tensorflow as tf
from ii_data_process import load_training_data, load_sample_training_data
from datetime import datetime

NUM_UNROLLINGS = 5
BATCH_SIZE = 30
NUM_STEPS = 7001
SUMMARY_FREQUENCY = 100
MODEL_PRE = ""
EMBEDDING_DIM = 400
HIDDEN_DIM = 128


def logprob(predictions, labels):
    """
    计算perplexity时用到。
    Log-probability of the true labels in a predicted batch.
    """
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    sample按照distribution的概率分布采样下标，这里的采样方式是针对离散的分布，相当于连续分布中求CDF。
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1


def sample(prediction, vocabulary_size):
    """Turn a (column) prediction into 1-hot encoded samples.
    根据sample_distribution采样得的下标值，转换成1-hot的样本
    """
    p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p


def random_distribution(vocabulary_size):
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b/np.sum(b, 1)[:, None]


class LSTM(object):
    """实现LSTM"""
    def __init__(self, hidden_dim, batch_size, num_unrollings, embedding_dim):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_unrollings = num_unrollings
        self.embedding_dim = embedding_dim
        self.graph = tf.Graph()
        self.load_data()
        self.build()

    def build(self):
        # 定义模型
        with self.graph.as_default():
            # Parameters:
            # Embedding layer
            with tf.name_scope("embedding"):
                W = tf.Variable(initial_value=W_value, name="W")
            # input to all gates
            x = tf.Variable(tf.truncated_normal([self.embedding_dim, self.hidden_dim * 4], -0.1, 0.1), name='x')
            # memory of all gates
            m = tf.Variable(tf.truncated_normal([self.hidden_dim, self.hidden_dim * 4], -0.1, 0.1), name='m')
            # biases all gates
            biases = tf.Variable(tf.zeros([1, self.hidden_dim * 4]))
            # Variables saving state across unrollings.
            saved_output = tf.Variable(tf.zeros([self.batch_size, self.hidden_dim]), trainable=False)
            saved_state = tf.Variable(tf.zeros([self.batch_size, self.hidden_dim]), trainable=False)
            # Classifier weights and biases.
            w = tf.Variable(tf.truncated_normal([self.hidden_dim, self.vocabulary_size], -0.1, 0.1))
            b = tf.Variable(tf.zeros([self.vocabulary_size]))
            keep_prob = tf.placeholder(tf.float32)

            # Definition of the cell computation.
            def lstm_cell(i, o, state):
                i = tf.nn.dropout(i, keep_prob)
                mult = tf.matmul(i, x) + tf.matmul(o, m) + biases
                input_gate = tf.sigmoid(mult[:, :self.hidden_dim])
                forget_gate = tf.sigmoid(mult[:, self.hidden_dim:self.hidden_dim * 2])
                update = mult[:, self.hidden_dim * 3:self.hidden_dim * 4]
                state = forget_gate * state + input_gate * tf.tanh(update)
                output_gate = tf.sigmoid(mult[:, self.hidden_dim * 3:])
                output = tf.nn.dropout(output_gate * tf.tanh(state), keep_prob)
                return output, state

            # Input data.
            self.train_inputs = list()
            self.train_labels = list()
            for _ in range(self.num_unrollings + 1):
                self.train_inputs.append(tf.placeholder(tf.float32, shape=[self.num_unrollings + 1, self.batch_size]))
                self.train_labels.append(tf.placeholder(tf.float32, shape=[self.num_unrollings + 1, self.batch_size]))

            # Unrolled LSTM loop.
            outputs = list()
            output = saved_output
            state = saved_state
            for i in self.train_inputs:
                # 计算每个lstm单元的输出和状态
                output, state = lstm_cell(tf.nn.embedding_lookup(embeddings, i),
                                          output, state)
                outputs.append(output)

            # State saving across unrollings.
            with tf.control_dependencies([saved_output.assign(output),
                                          saved_state.assign(state)]):
                # Classifier.
                logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits, tf.concat(0, self.train_labels)))

            # Optimizer.
            global_step = tf.Variable(0)
            self.learning_rate = tf.train.exponential_decay(
                10.0, global_step, 5000, 0.1, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
            self.optimizer = optimizer.apply_gradients(
                zip(gradients, v), global_step=global_step)

            # Predictions.
            self.train_prediction = tf.nn.softmax(logits)

            # Sampling
            sample_input = tf.placeholder(tf.float32, shape=[1, self.vocabulary_size])
            saved_sample_output = tf.Variable(tf.zeros([1, self.hidden_dim]))
            saved_sample_state = tf.Variable(tf.zeros([1, self.hidden_dim]))
            self.reset_sample_state = tf.group(
                saved_sample_output.assign(tf.zeros([1, self.hidden_dim])),
                saved_sample_state.assign(tf.zeros([1, self.hidden_dim])))
            sample_output, sample_state = lstm_cell(
                sample_input, saved_sample_output, saved_sample_state)
            with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                          saved_sample_state.assign(sample_state)]):
                self.sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

    def load_data(self):
        # Load data
        self.X_train, self.y_train, self.char_to_index, self.index_to_char = load_sample_training_data(1)
        print "X_train: {}, {} y_train: {}, {} ".format(self.X_train[0], [self.index_to_char[x] for x in self.X_train[0]],
                                                        self.y_train[0], [self.index_to_char[y] for y in self.y_train[0]])
        self.vocabulary_size = len(self.char_to_index.keys())
        self.bg = BatchGenerator(X_value=self.X_train, Y_value=self.y_train, batch_size=BATCH_SIZE,
                                 num_unrollings=NUM_UNROLLINGS, vocabulary_size=self.vocabulary_size,
                                 char_to_index=self.char_to_index)

    def train(self):
        #self.load_data()

        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            # 有预训练的模型，则加载
            if MODEL_PRE:
                saver.restore(session, MODEL_PRE)
            tf.initialize_all_variables().run()
            print('Initialized')
            mean_loss = 0
            for step in range(NUM_STEPS):
                X_batchs, Y_batchs = self.bg.next()
                feed_dict = dict()
                for i in range(self.num_unrollings + 1):
                    feed_dict[self.train_inputs[i]] = X_batchs[i]
                    feed_dict[self.train_labels[i]] = Y_batchs[i]
                _, l, predictions, lr = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.learning_rate], feed_dict=feed_dict)
                mean_loss += l
                if step % SUMMARY_FREQUENCY == 0:
                    if step > 0:
                        mean_loss = mean_loss / SUMMARY_FREQUENCY
                    # The mean loss is an estimate of the loss over the last few batches.
                    print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                    mean_loss = 0
                    print('Minibatch perplexity: %.2f' % float(
                        np.exp(logprob(predictions, np.asarray(Y_batch)))))
                    if step % (SUMMARY_FREQUENCY * 10) == 0:
                        print('=' * 80)
                        ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
                        model_output_file = "model/gru_c2v/GRU-%s-%s-%s-%s.dat" % (ts, self.vocabulary_size, EMBEDDING_DIM, HIDDEN_DIM)
                        saver.save(session, model_output_file)
                        print("model saved to {}".format(model_output_file))


class BatchGenerator(object):
    """Batch 生成器"""
    def __init__(self, X_value, Y_value, batch_size,
                 num_unrollings, vocabulary_size, char_to_index):
        self.X_value = X_value
        # print self.X_value[:20]
        # print self.X_value.shape
        self.Y_value = Y_value
        self.data_len = len(X_value)
        self.batch_size = batch_size
        self.num_unrollings = num_unrollings
        self.vocabulary_size = vocabulary_size
        self.char_to_index = char_to_index
        self.start = 0
        self.end = batch_size - 1

    def next(self):
        X_all = self.X_value[range(self.start, self.end + 1)]
        Y_all = self.Y_value[range(self.start, self.end + 1)]
        X_all = [x + list(np.zeros(self.num_unrollings - len(x), dtype=int)) for x in X_all if len(x) != self.num_unrollings]
        Y_all = [y + list(np.zeros(self.num_unrollings - len(y), dtype=int)) for y in Y_all if len(y) != self.num_unrollings]
        X_batchs = list()
        Y_batchs = list()
        for step in range(self.num_unrollings):
            X_batch = list()
            # X_batch = np.zeros(shape=(self.batch_size, self.vocabulary_size), dtype=np.float)
            Y_batch = np.zeros(shape=(self.batch_size, self.vocabulary_size), dtype=np.float)
            for b in range(self.batch_size):
                X_batch.append(X_all[b][step])
                Y_batch[b, X_all[b][step]] = 1.0
            X_batchs.append(X_batch)
            Y_batchs.append(Y_batch)
        self.start = (self.end + 1) % self.data_len
        self.end = (self.end + self.batch_size - 1) % self.data_len
        return X_batchs, Y_batchs
    # def next(self):
    #     """产生下一个batch数据"""
    #     graph = tf.Graph()
    #     with graph.as_default():
    #         X = tf.placeholder(tf.float32, shape=[self.data_len, None])
    #         Y = tf.placeholder(tf.float32, shape=[self.data_len, None])
    #
    #         # Batch the variable length tensor with dynamic padding
    #         batched_data = tf.train.batch(
    #             tensors=[X],
    #             batch_size=self.batch_size,
    #             dynamic_pad=True,
    #             name="x_y_batch"
    #         )
    #
    #     with tf.Session(graph=graph) as sess:
    #         feed_dict = {X: self.X_value, Y: self.Y_value}
    #         print sess.run(batched_data, feed_dict=feed_dict)


if __name__ == '__main__':
    # Load data
    X_train, y_train, char_to_index, index_to_char = load_sample_training_data(1)
    print "X_train: {}, {} y_train: {}, {} ".format(X_train[0], [index_to_char[x] for x in X_train[0]],
                                                   y_train[0], [index_to_char[y] for y in y_train[0]])
    vocabulary_size = len(char_to_index.keys())
    bg = BatchGenerator(X_value=X_train, Y_value=y_train, batch_size=BATCH_SIZE,
                             num_unrollings=NUM_UNROLLINGS, vocabulary_size=vocabulary_size,
                             char_to_index=char_to_index)
    print bg.next()

    #model = LSTM(batch_size=BATCH_SIZE, hidden_dim=HIDDEN_DIM, num_unrollings=NUM_UNROLLINGS, embedding_dim=EMBEDDING_DIM)

    #model.train()

