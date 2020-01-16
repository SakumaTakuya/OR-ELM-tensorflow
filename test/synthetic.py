#%%
import argparse
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

sys.path.append(base_path)

from module import eval
from module import or_elm
from module import preprocessing

from module import rc_os_elm_np

from train_predict import Evaluation


if __name__ == "__main__":
    norm_name = 'walk_norm_0'
    data_name = 'cd_run_walk_stop_back_short_0'

    normal_data_base = np.loadtxt('C:\\Users\\Lab\\Documents\\sakuma\\unity\\AxisNeuronDataCollection\\Assets\\csvs\\Sequential\\data_' + norm_name + '.csv', delimiter=',')
    anormaly_data_base = np.loadtxt('C:\\Users\\Lab\\Documents\\sakuma\\unity\\AxisNeuronDataCollection\\Assets\\csvs\\Sequential\\data_' + data_name + '.csv', delimiter=',')
    label = np.loadtxt('C:\\Users\\Lab\\Documents\\sakuma\\unity\\AxisNeuronDataCollection\\Assets\\csvs\\Sequential\\label_' + data_name + '.csv', delimiter=',')

    in_shape = normal_data_base.shape[1]
    print(in_shape)
    look_back = 1
    pred_length = 1
    forget_fact = 1
    rc_forget = 0.999
    leak_rate = 0.5
    regularizer = 1 / np.e ** 5
    unit = 28 // 4
    ada_forget = 0.9999
    delay = 0
    outlier_rate = 2

    normal_data, normal_nex = preprocessing.create_subseq(normal_data_base, look_back, pred_length)
    anormaly_data, anormaly_nex = preprocessing.create_subseq(anormaly_data_base, look_back, pred_length)

    normal_data = np.array(normal_data)
    normal_nex =  np.array(normal_nex)
    anormaly_data = np.array(anormaly_data)
    anormaly_nex = np.array(anormaly_nex)

    normal_data = normal_data.reshape((-1, in_shape * look_back))
    normal_nex = normal_nex.reshape((-1, in_shape * pred_length))

#%%
    mean_score = eval.AdaptiveStatisticCalculator()
    evaluator = Evaluation(None, ada_forget)
    for i in range(1):
        rc_elm = rc_os_elm_np.RLS_ESN_Delay(
            hi_shape=unit,
            init_x=normal_data[:1],
            init_y=normal_nex[:1],
            delay=delay,
            outlier_rate=outlier_rate,
            forget_fact=forget_fact,
            leak_rate=leak_rate,
            gen_w_res=lambda x: rc_os_elm_np.generate_reservoir_weight(x, rc_forget))

        evaluator.model = rc_elm
        evaluator.scores.clear()

        # evaluator.train(normal_data, normal_nex, can_save=False)
        print(anormaly_data.shape, anormaly_nex.shape)
        evaluator.train(anormaly_data, anormaly_nex,can_save=True)

        mean_score.update(np.array(evaluator.scores))
    
    np.savetxt(
        'C:\\Users\\Lab\\Documents\\sakuma\\unity\\AxisNeuronDataCollection\\Assets\\csvs\\Sequential\\predtest.csv',
        np.array(evaluator.preds),
        delimiter=',')
    evaluator.scores = mean_score.mean
    evaluator.save_score(
        os.path.join(
            base_path,
            'result',
            'synth',
            f'{data_name}_look{look_back}_pred{pred_length}_adaf{ada_forget}_auc{evaluator.max_auc}.csv'))

#%%
    evaluator.label = label[:len(evaluator.scores)]
    # evaluator.scores = evaluator.scores[:len(label)]
    evaluator.show_fig(can_show_auc=False)

# %%
