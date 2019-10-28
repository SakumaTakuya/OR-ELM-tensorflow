#%%
import numpy as np
import tensorflow as tf

def calcu_mse(value, predict, variance=0.1):
    mse_value = [(v - p)**2 / variance for v, p in zip(value, predict)]
    return  np.mean(np.array(mse_value), axis=2)

def create_subseq(ts, look_back, pred_length):
    sub_seq, next_values = [], []
    for i in range(len(ts)-look_back-pred_length):  
        sub_seq.append(ts[i:i+look_back])
        next_values.append(ts[i+look_back:i+look_back+pred_length])
    return sub_seq, next_values

lock_back = 3
pred_length = 1

#%%
if __name__ == '__main__':
    model = tf.keras.models.load_model('/home/sakuma/work/simple_gru.h5')

#%%
    normal_data = np.loadtxt('/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_5.csv', delimiter=',')
    anormaly_data = np.loadtxt('/home/sakuma/work/python/OR-ELM-tensorflow/data/data_0.csv', delimiter=',')

#%%
    normal_data, normal_nex = create_subseq(normal_data, lock_back, pred_length)
    anormaly_data, anormaly_nex = create_subseq(anormaly_data, lock_back, pred_length)

#%%
    normal_data = np.array(normal_data)[np.newaxis]
    anormaly_data = np.array(anormaly_data)[np.newaxis]

#%%
    normal_pred = model.predict(normal_data)
    anormaly_pred = model.predict(anormaly_data)

#%%
    normal_mse = calcu_mse(normal_nex, normal_pred[0])
    anormaly_mse = calcu_mse(anormaly_nex, anormaly_pred[0])

# %%
import matplotlib.pyplot as plt

