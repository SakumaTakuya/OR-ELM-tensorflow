#%%
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy import stats
import sys
import tqdm 

base_path = 'C:\\Users\\Lab\\Documents\\sakuma\\unity\\AxisNeuronDataCollection\\Assets\\python\\OR-ELM-tensorflow\\'
data_path = 'C:\\Users\\Lab\\Documents\\sakuma\\dataset\\On-body Localization of Wearable Devices An Investigation of Position-Aware Activity Recognition'

sys.path.append(base_path)
sys.path.append(base_path + '\\test')

from module import eval
from module import or_elm
from module import preprocessing
from module import rc_os_elm_np

from train_predict import fit_score_offset

#%%
base = 'C:\\Users\\Lab\\Documents\\sakuma\\unity\\AxisNeuronDataCollection\\Assets\\python\\OR-ELM-tensorflow\\result\\synth'
# target = 'for_adaforget_foget_0.9999_rc0.9999'
loadtxt = lambda path: np.loadtxt(
                        os.path.join(
                            base,
                            # target,
                            path),delimiter=',')

# %%

if __name__ == '__main__':
    data_name = 'cd_run_walk_stop_back_short_0'
    label = np.loadtxt('C:\\Users\\Lab\\Documents\\sakuma\\unity\\AxisNeuronDataCollection\\Assets\\csvs\\Sequential\\label_' + data_name + '.csv', delimiter=',')

    rls09999 = loadtxt('cd_run_walk_stop_back_short_0_look1_pred1_adaf0.999581_auc0.csv')
    rls1 = loadtxt('cd_run_walk_stop_back_short_0_look1_pred1_adaf0.9999_auc0_forget1.csv')
    lstm = loadtxt('mse_synth_f0.9999.csv')

#%%
    fig = plt.figure()
    sub1 = fig.add_subplot(311)
    sub2 = fig.add_subplot(312, sharex=sub1, sharey=sub1)
    sub3 = fig.add_subplot(313, sharex=sub1, sharey=sub1)

    eval.plot_mse(
        lstm,
        label,
        can_flush=False,
        label=None,
        xlim=[0, len(label)],
        ylim=[0, stats.chi2.interval(0.9999, 1)[1]],
        sub=sub1)
    eval.plot_mse(rls1, label, label=None,sub=sub2, can_flush=False)
    eval.plot_mse(rls09999, label, label=None, sub=sub3, can_flush=False)

    sub1.set_title('LSTM')
    sub2.set_title('RLS-ESN(忘却率=1.0000)')
    sub3.set_title('RLS-ESN(忘却率=0.9999)')
    sub2.set_ylabel('異常度')
    plt.xlabel('時間ステップ')
    fig.tight_layout()
    plt.show()