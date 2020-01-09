#%%
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from scipy import signal
from sklearn import metrics
import sys
# import tensorflow as tf
import tqdm 

base_path = 'C:\\Users\\Lab\\Documents\\sakuma\\unity\\AxisNeuronDataCollection\\Assets\\python\\OR-ELM-tensorflow\\'
data_path = 'C:\\Users\\Lab\\Documents\\sakuma\\dataset\\On-body Localization of Wearable Devices An Investigation of Position-Aware Activity Recognition'

sys.path.append(base_path)

from module import eval
from module import or_elm
from module import preprocessing

from module import rc_os_elm_np

from train_predict import Evaluation

data_type = 'running'

def get_data(subject):
    base = os.path.join(data_path, subject)

    data_base = []
    for p in tqdm.tqdm(glob.glob(os.path.join(base, f'data_all_{data_type}_*.csv')), desc='load csv'):
        d = np.loadtxt(
            p,
            delimiter=',')
        d = d.reshape(len(d), -1)
        data_base.append(d)

    datas = []
    nexts = []
    for data in tqdm.tqdm(data_base, desc='create data'):
        d, n = preprocessing.create_subseq(data, look_back, pred_length)
        d = np.array(d).reshape((-1, in_shape * look_back))
        n = np.array(n).reshape((-1, in_shape * pred_length))

        datas.append(d)
        nexts.append(n)

    return np.array(datas).squeeze(), np.array(nexts).squeeze()

if __name__ == "__main__":
    norm = [
        '1',
        '10',
        '11', 
        '12', 
    ]

    anom = '13'

    in_shape = 28
    look_back = 1
    pred_length = 1
    unit = pred_length * 1 * in_shape
    forget_fact = 0.999
    rc_forget = 0.9999
    leak_rate = 0.5
    regularizer = 1 / np.e ** 5
    unit = pred_length * 1 * in_shape
    ada_forget = 0.99

    delay = 0
    outlier_rate = 2

    normal_data = []
    normal_nex = []
    for subject in norm:
        datas, nexts = get_data(subject)
        normal_data.append(datas)
        normal_nex.append(nexts)

        print(subject, datas.shape, nexts.shape)

    normal_data = np.concatenate(normal_data)
    normal_nex = np.concatenate(normal_nex)

    anormaly_data, anormaly_nex = get_data(anom)
    label = np.loadtxt(os.path.join(data_path, anom, 'ground_anom.csv'))

    mean_score = eval.AdaptiveStatisticCalculator()
    evaluator = Evaluation(None, ada_forget)
    for i in range(1):
        elm = rc_os_elm_np.RLS_ESN_Delay(
            hi_shape=unit,
            forget_fact=forget_fact,
            leak_rate=leak_rate,
            regularizer=regularizer,
            delay=delay,
            outlier_rate=outlier_rate,
            init_x=normal_data[:1],
            init_y=normal_nex[:1],
            gen_w_res=lambda x: rc_os_elm_np.generate_reservoir_weight(x, rc_forget))
        
        evaluator.model = elm
        evaluator.scores.clear()

        evaluator.train(normal_data, normal_nex, can_save=False)
        evaluator.train(anormaly_data, anormaly_nex,can_save=True)

        mean_score.update(np.array(evaluator.scores))

    evaluator.scores = mean_score.mean
    evaluator.fit_score_offset(label)
    evaluator.save_score(
        os.path.join(
            base_path,
            'result',
            'real',
            f'{data_type}_{anom}_look{look_back}_pred{pred_length}_adaf{ada_forget}_auc{evaluator.max_auc}.csv'))
    evaluator.show_fig()

