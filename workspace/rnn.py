#%%
import numpy as np
import tqdm
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.append('/home/sakuma/work/python/OR-ELM-tensorflow/')

from layers import dense

#%%
class MinimalRNNCell(layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

#%%
# keras のRNNよくわからんのでtensorflowで書き直す
class Rnn:
    def __init__(
        self, 
        in_shape, 
        hidden_shape, 
        out_shape, 
        time_step=3,
        pred_step=1,
        loss='mean_squared_error',
        optimizer='sgd'):
        self.inputs = tf.keras.Input((None, time_step, in_shape))
        # self.inputs = tf.keras.Input((None, time_step * in_shape))
        self.flat = tf.keras.layers.Reshape((-1, time_step * in_shape))(self.inputs)
        self.hidden = layers.GRU(hidden_shape * pred_step,
                                 return_sequences=True, # return the last output in the output sequence, or the full sequence
                                )(self.flat)
        self.dense = tf.keras.layers.Dense(pred_step * out_shape)(self.hidden)
        self.output = tf.keras.layers.Reshape((-1, pred_step, out_shape))(self.dense)

        # self.output = K.stack([
        #     layers.Dense(out_shape)(self.hidden) for i in range(pred_step)],
        #     axis=2)
        # self.output = tf.keras.layers.Dense(pred_step * out_shape)(self.hidden)

        self.model = tf.keras.models.Model(
            inputs=self.inputs, outputs=self.output)

        self.model.summary()

        self.model.compile(
            loss=loss, 
            optimizer=optimizer,
            metrics=['mse'])

    def train(self, x, y, val_x, val_y, batch_size, epochs):
        self.model.fit(
            x, 
            y, 
            validation_data=(val_x, val_y),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[tf.keras.callbacks.EarlyStopping()])

    def save(self, path):
        self.model.save(path, overwrite=False)

#%%
# class RnnTf:
#     def __init__(self, in_shape, hidden_shape, out_shape, time_step, pred_step):
#         def rnn(x, hidden, act=tf.nn.tanh, name='rnn'):
#             _, h = [ i.value for i in x.shape ]
#             with tf.name_scope(name):
#                 w = tf.Variable(
#                     tf.truncated_normal([h, hidden]))
#                 v = tf.Variable(
#                     tf.truncated_normal([hidden, hidden]))
#                 b = tf.Variable(
#                     tf.zeros(hidden))

#                 x_1 = tf.Variable(
#                     initial_value=tf.zeros([1, hidden]),
#                     shape=[None, hidden],
#                     name='x_1')

#                 return tf.assign(x_1,
#                     act(tf.matmul(x, w) + tf.matmul(x_1, v) + b))

#         self.input = tf.placeholder(
#             tf.float32, shape=[None, time_step, in_shape])
#         self.true = tf.placeholder(
#             tf.float32, [None, pred_step, out_shape])

#         self.flat = tf.reshape(self.input, [-1, time_step*in_shape])

#         self.h = rnn(
#             self.flat, 
#             hidden_shape)

#         self.o = tf.stack([
#             dense(self.h, out_shape, act=None) for i in range(pred_step)],
#             axis=1)
        
#         self.cost = tf.reduce_mean(
#             tf.square(self.true - self.o))

#         self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

#     def train(self, xs, ys, sess):
#         """
#         time * batch_size * lock_back * in_shape
#         """
#         for x, y in zip(xs, ys):
#             sess.run(self.train_op, feed_dict={
#                 self.input : x,
#                 self.true  : y})

#%%
if __name__ == '__main__':
    lock_back = 3
    pred_length = 1
    in_shape = 455
    rnn = Rnn(
        in_shape, 
        in_shape // 4, 
        in_shape,
        pred_step=pred_length,
        time_step=lock_back,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

#%%
    def create_subseq(ts, look_back, pred_length):
        sub_seq, next_values = [], []
        for i in range(len(ts)-look_back-pred_length):  
            sub_seq.append(ts[i:i+look_back])
            next_values.append(ts[i+look_back:i+look_back+pred_length])
        return sub_seq, next_values

    data_path = [
        '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_0.csv',
        '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_1.csv',
        '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_2.csv',
        '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_3.csv',
        '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_4.csv',
        # '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_5.csv',
        # '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_6.csv',
        # '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_7.csv',
    ]

    sub_datas = [] # batch_size * time * lock_back * in_shape
    nex_datas = [] # batch_size * time * pred_length * in_shape
    for p in tqdm.tqdm(data_path):
        sub, nex = create_subseq(
            np.loadtxt(p, delimiter=',')[:1092], # time * channel
            lock_back, 
            pred_length)
        sub_datas.append(sub)
        nex_datas.append(nex)

#%%
    # time * batch_size * lock_back * in_shape
    sub_datas = np.array(sub_datas) #.transpose(1,0,2,3) 
    nex_datas = np.array(nex_datas) #.transpose(1,0,2,3)

#%%
    print(sub_datas[:4].shape)
    print(nex_datas[:4].shape)

#%%
    rnn.train(
        sub_datas[:4], #.reshape((8, -1, 4550)), 
        nex_datas[:4], #.reshape((8, -1, 1365)), 
        sub_datas[4:5],
        nex_datas[4:5],
        4, 2000)

    rnn.save("simple_gru.h5")

#%% for tf
    # epochs = 200
    # init_op = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     for epoch in range(epochs):
    #         rnn.train(
    #             sub_datas, 
    #             nex_datas,
    #             sess)

# %%
# if __name__ == '__main__':
#     import glob

#     def create_subseq(ts, look_back, pred_length):
#         sub_seq, next_values = [], []
#         for i in range(len(ts)-look_back-pred_length):  
#             sub_seq.append(ts[i:i+look_back])
#             next_values.append(ts[i+look_back:i+look_back+pred_length].T[0])
#         return sub_seq, next_values

#     def load_data():
#         for path in glob.glob('/home/sakuma/work/python/OR-ELM-tensorflow/data/x_pose_*.npy'):
#             data = np.load(path)
#             for i in range(len(data)):

