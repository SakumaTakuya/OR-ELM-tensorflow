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


class Evaluation:
    def __init__(self, model, ada_forget):
        self.model = model
        self.adaMse = eval.AdaptiveWeightedStatisticCalculator(ada_forget)

        self.label = None
        self.scores = []
        self.max_auc = 0
        self.max_offset = -999999

    def train(self, inputs, outputs, can_save=False):
        for data in [zip(inputs, outputs)]:
            for x, y in tqdm.tqdm(data):
                x = x.reshape([1, -1])
                y = y.reshape([1, -1])
                self.model.train(x, y)
                out = self.model.output()

                assert not np.any(np.isnan(out))

                mse = np.mean((y - out) ** 2)
                self.adaMse.update(mse)

                if can_save:  
                    self.scores.append(self.adaMse.get_hotelling_stat(mse))
    
    def fit_score_offset(self, label):
        def move_label(label, i):
            return  np.concatenate([np.zeros(i), label[:-i]]) \
                    if i > 0 else \
                    np.concatenate([label[-i:], np.zeros(-i)])

        self.scores = np.array(self.scores)
        diff = self.scores.shape[0] - label.shape[0]
        if diff > 0:
            label = np.concatenate([np.zeros(diff), label])
        elif diff < 0:
            self.scores = np.concatenate([np.zeros(-diff), self.scores])

        for i in range(-2048, 2048):
            l = move_label(label, i)

            fpr, tpr, _ = metrics.roc_curve(
                np.where(l == 0, 0, 1), 
                self.scores)

            auc = metrics.auc(fpr, tpr)

            if auc > self.max_auc:
                self.max_auc = auc
                self.max_offset = i

        self.label = move_label(label, self.max_offset)

    def save_score(self, path):
        np.savetxt(path, self.scores, delimiter=',')

    def show_fig(self, auc=False, mse=True):
        eval.plot_auc(
            self.scores,
            self.label,
            can_flush=False)
        plt.show()
        
        eval.plot_mse(
            self.scores,
            self.label,
            can_flush=False,
            xlim=[0, len(self.scores)],
            ylim=[0, np.sort(self.scores[len(self.scores)//4:])[-1]*1.1])
        plt.show()


if __name__ == "__main__":
    norm_name = 'cd_run_walk_stop_back_0'
    data_name = 'cd_run_walk_stop_back_0'

    normal_data_base = np.loadtxt('C:\\Users\\Lab\\Documents\\sakuma\\unity\\AxisNeuronDataCollection\\Assets\\csvs\\Sequential\\data_' + norm_name + '.csv', delimiter=',')
    anormaly_data_base = np.loadtxt('C:\\Users\\Lab\\Documents\\sakuma\\unity\\AxisNeuronDataCollection\\Assets\\csvs\\Sequential\\data_' + data_name + '.csv', delimiter=',')
    label = np.loadtxt('C:\\Users\\Lab\\Documents\\sakuma\\unity\\AxisNeuronDataCollection\\Assets\\csvs\\Sequential\\label_' + data_name + '.csv', delimiter=',')

    in_shape = normal_data_base.shape[1]
    look_back = 1
    pred_length = 1
    forget_fact = 0.999
    rc_forget = 0.999
    leak_rate = 0.5
    regularizer = 1 / np.e ** 5
    unit = pred_length * 1 * in_shape
    ada_forget = 0.999

    normal_data, normal_nex = preprocessing.create_subseq(normal_data_base, look_back, pred_length)
    anormaly_data, anormaly_nex = preprocessing.create_subseq(anormaly_data_base, look_back, pred_length)

    normal_data = np.array(normal_data)
    normal_nex =  np.array(normal_nex)
    anormaly_data = np.array(anormaly_data)
    anormaly_nex = np.array(anormaly_nex)

    normal_data = normal_data.reshape((-1, in_shape * look_back))
    normal_nex = normal_nex.reshape((-1, in_shape * pred_length))

#%%
    rc_elm = rc_os_elm_np.RC_OS_ELM(
        hi_shape=unit,
        init_x=normal_data[:1],
        init_y=normal_nex[:1],
        forget_fact=forget_fact,
        leak_rate=leak_rate)

    evaluator = Evaluation(
        rc_elm,
        ada_forget)

    evaluator.train(normal_data, normal_nex, can_save=False)
    evaluator.train(anormaly_data, anormaly_nex,can_save=True)
    
    evaluator.save_score(
        os.path.join(
            base_path,
            'result',
            f'{data_name}_look{look_back}_pred{pred_length}_adaf{ada_forget}.csv'))

    evaluator.label = label[:len(evaluator.scores)]
    # evaluator.scores = evaluator.scores[:len(label)]
    evaluator.show_fig()