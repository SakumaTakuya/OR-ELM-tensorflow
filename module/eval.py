import matplotlib.pyplot as plt
from matplotlib import collections 
import numpy as np

def calcu_mse(value, predict, variance=0.1):
    mse_value = [(v - p)**2 / variance for v, p in zip(value, predict)]
    return  np.mean(np.array(mse_value), axis=2)

def plot(normal_mse, anormaly_mse, anormaly_label, cut=0):
    normal_mse = normal_mse[cut:]
    anormaly_mse = anormaly_mse[cut:]
    anormaly_label = anormaly_label[cut:]
    fig = plt.figure()
    sub = fig.add_subplot(111)

    clt = collections.BrokenBarHCollection.span_where(
        np.arange(anormaly_label.shape[0]), 
        ymin=0, 
        ymax=max(
            np.max(normal_mse),
            np.max(anormaly_mse)
        ), 
        where=anormaly_label>0, 
        facecolor='red', alpha=0.5)
    sub.add_collection(clt)
    sub.plot(normal_mse, label='normal mse')
    sub.plot(anormaly_mse, label='anormaly mse')
    plt.legend()
    fig.show()