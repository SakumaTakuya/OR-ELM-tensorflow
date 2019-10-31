"""
参考:
https://www.kabuku.co.jp/developers/time_series_anomaly_detect_deep_learning
"""
import matplotlib.pyplot as plt
from matplotlib import collections 
import numpy as np
import os
from sklearn import metrics


def calcu_mse(value, predict, variance=0.1):
    mse_value = [(v - p)**2 / variance for v, p in zip(value, predict)]
    return  np.mean(np.array(mse_value), axis=2)

def plot_mse(anormaly_mse, anormaly_label, cut=0, save_path=None):
    anormaly_mse = anormaly_mse[cut:]
    anormaly_label = anormaly_label[cut:]
    fig = plt.figure()
    sub = fig.add_subplot(111)

    clt = collections.BrokenBarHCollection.span_where(
        np.arange(anormaly_label.shape[0]), 
        ymin=0, 
        ymax=np.max(anormaly_mse), 
        where=anormaly_label>0, 
        facecolor='red', alpha=0.5)
    sub.add_collection(clt)
    sub.set_ylim([0, 1500])
    sub.plot(anormaly_mse, label='mse')
    sub.legend()

    save_plot(save_path)

    fig.show()
    plt.clf()
    

def plot_auc(anormaly_mse, anormaly_label, save_path=None):
    fpr, tpr, _ = metrics.roc_curve(
        np.where(anormaly_label == 0, 0, 1), 
        anormaly_mse)

    auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()

    save_plot(save_path)

    plt.show()
    plt.clf()

    
def save_plot(save_path):
    if save_path is None:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    
    

