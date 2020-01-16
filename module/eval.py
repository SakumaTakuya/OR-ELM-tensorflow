"""
参考:
https://www.kabuku.co.jp/developers/time_series_anomaly_detect_deep_learning
"""

import math
import matplotlib.pyplot as plt
from matplotlib import collections 
from matplotlib import colors
import numpy as np
import os
from sklearn import metrics

class AdaptiveStatisticCalculator:
    def __init__(self):
        self.mean = 0
        self.mean2 = 0
        self.var = 0
        self.iter = 0

    def update(self, value):
        prev_iter = self.iter
        prev_mean2 = self.mean2

        self.iter += 1        
        self.mean = (prev_iter * self.mean + value) / self.iter
        self.mean2 = self.mean ** 2
        self.var = (prev_iter * (self.var + prev_mean2) + value**2) / self.iter - self.mean2

    def get_hotelling_stat(self, value):
        return  (value - self.mean) ** 2 / self.var \
                if np.all(self.var != 0) else \
                (value - self.mean) ** 2

class AdaptiveWeightedStatisticCalculator:
    def __init__(self, forget_factor):
        self.mean = 0
        self.mean2 = 0
        self.var = 0
        self.forget_factor = forget_factor
        self.forget_sum = 1

    def update(self, value):
        prev_forget = self.forget_sum
        prev_mean2 = self.mean2

        self.forget_sum *= self.forget_factor
        self.forget_sum += 1

        self.mean = (self.forget_factor * prev_forget * self.mean + value) / self.forget_sum
        self.mean2 = self.mean ** 2
        self.var = (prev_forget * self.forget_factor * (self.var + prev_mean2) + value**2) / self.forget_sum - self.mean2

    def get_hotelling_stat(self, value):
        return  (value - self.mean) ** 2 / self.var \
                if np.all(self.var != 0) else \
                (value - self.mean) ** 2 

class ExponentiallyWeightedStatisticCalculator:
    def __init__(self, alpha, init_mean=0, init_var=0):
        self.alpha = alpha
        self.mean = init_mean
        self.var = init_var

    def update(self, value):
        diff = value - self.mean
        incr = self.alpha * diff
        self.mean = self.mean + incr
        self.var = (1 - self.alpha) * (self.var + diff * incr)
    
    def get_hotelling_stat(self, value):
        return  (value - self.mean) ** 2 / self.var \
                if np.all(self.var != 0) else \
                (value - self.mean) ** 2 


class AdaptiveWeightedStatisticCalculator2:
    def __init__(self, forget_factor):
        self.forget_factor = forget_factor
        self.weighted_sum = 0
        self.forget_sum = 0
        self.mean = 0

    def update(self, value):
        self.weighted_sum = self.forget_factor * self.weighted_sum + value
        
        self.forget_sum *= self.forget_factor
        self.forget_sum += 1

        self.mean = self.weighted_sum / self.forget_sum


# class AdaptiveStatisticCalculator:
#     def __init__(self):
#         self.mean = 0
#         self.mean2 = 0
#         self.var = 0
#         self.iter = 0

#     def next_score(self, value, predict, outlier=None):
#         prev_iter = self.iter
#         prev_mean2 = self.mean2

#         mse = np.mean((value - predict)**2)

#         if outlier is not None:
#             stat = self.__get_stat(mse)
#             if stat > outlier:
#                 return stat
        
#         self.iter += 1        
#         self.mean = (prev_iter * self.mean + mse) / self.iter
#         self.mean2 = self.mean ** 2
#         self.var = (prev_iter * (self.var + prev_mean2) + mse**2) / self.iter - self.mean2

#         return self.__get_stat(mse)

#     def __get_stat(self, mse):
#         return (mse - self.mean) ** 2 / self.var if self.var != 0 else (mse - self.mean) ** 2


def calcu_mse(value, predict, variance=0.1):
    """
        value.shape (data_num, pred, data shape)
    """
    mse_value = [(v - p)**2 / variance for v, p in zip(value, predict)]
    mse_value = np.array(mse_value)
    return  np.mean(mse_value, axis=tuple(range(1, len(mse_value.shape))))

def plot_mse(anormaly_mse, anormaly_label=None, cut=0, save_path=None, can_flush=True, label='mse', xlim=None, ylim=None, sub=plt.figure().add_subplot(111)):
    hsv2rgb = np.frompyfunc(lambda x : colors.hsv_to_rgb([x, 0.6, 1]), 1, 1)
    
    anormaly_mse = anormaly_mse[cut:]

    if label is not None:
        sub.plot(anormaly_mse, label=label)
        sub.legend()
    else:
        sub.plot(anormaly_mse)        

    if xlim is not None:
        sub.set_xlim(xlim)

    if ylim is not None:
        sub.set_ylim(ylim)

    if anormaly_label is not None:
        anormaly_label = np.sin(anormaly_label[cut:] / (max(anormaly_label)) * np.pi / 4)
        # anormaly_label = np.convolve(anormaly_label, np.ones(512)/512)
        anormaly_label = np.array(list(hsv2rgb(anormaly_label)))
        anormaly_label = anormaly_label[np.newaxis]

        x_lim = sub.get_xlim()
        y_lim = sub.get_ylim()
        sub.imshow(
            anormaly_label[:len(anormaly_mse)], 
            extent=[*x_lim, *y_lim], aspect='auto', alpha=0.5)


    save_plot(save_path)

    if can_flush:
        plt.clf()
    

def plot_auc(anormaly_mse, anormaly_label, save_path=None, can_flush=True, xlim=None):
    fpr, tpr, _ = metrics.roc_curve(
        np.where(anormaly_label == 0, 0, 1), 
        anormaly_mse)

    auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)

    save_plot(save_path)

    # plt.show()
    if can_flush:
        plt.clf()

    
def save_plot(save_path):
    if save_path is None:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

def get_save_path(*path):
    path = os.path.join(*path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def plot(pred, nex, label, path_func, can_plot_mse=True, can_plot_auc=True, cut=0):

    print(path_func('xxxxx'))
    anorm_mse = calcu_mse(nex, pred)

    if can_plot_mse:
        plot_mse(anorm_mse, label, cut=cut, save_path=path_func('anorm_mse'))

    if can_plot_auc:
        plot_auc(anorm_mse, label[-len(anorm_mse):], save_path=path_func('auc'))
    
    

