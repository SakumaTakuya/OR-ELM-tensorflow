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
base = 'C:\\Users\\Lab\\Documents\\sakuma\\unity\\AxisNeuronDataCollection\\Assets\\python\\OR-ELM-tensorflow\\result\\real'
target = 'for_adaforget_foget_0.9999_rc0.9999'
loadtxt = lambda path: np.loadtxt(
                        os.path.join(
                            base,
                            target,
                            path),delimiter=',')

if __name__ == '__main__':
    label = np.loadtxt(os.path.join(
        data_path, '13', 'ground_anom.csv'))
    data1 = loadtxt('running_13_look1_pred1_adaf1_auc0.8055098235643797.csv')
    data09999 = loadtxt('running_13_look1_pred1_adaf0.9999_auc0.9047622812403476.csv')
    data0999 = loadtxt('running_13_look1_pred1_adaf0.999_auc0.8083191457861595.csv')

    label = fit_score_offset(label, data1)[0]
#%%
    fig = plt.figure()
    sub1 = fig.add_subplot(311)
    # sub2 = fig.add_subplot(312, sharex=sub1, sharey=sub1)
    sub2 = fig.add_subplot(312, sharex=sub1)
    # sub3 = fig.add_subplot(313, sharex=sub1, sharey=sub1)
    sub3 = fig.add_subplot(313, sharex=sub1)

    eval.plot_mse(
        data1,
        label,
        can_flush=False,
        label=None,
        xlim=[0, len(label)],
        ylim=[0, np.max(data1[len(data1)//4:])*1.1],
        sub=sub1)
    # eval.plot_mse(data09999, label, label=None,sub=sub2, can_flush=False)
    eval.plot_mse(
        data09999, 
        label, 
        label=None,
        ylim=[0, np.max(data09999[len(data09999)//4:])*1.1],
        sub=sub2, 
        can_flush=False)
    # eval.plot_mse(data0999, label, label=None, sub=sub3, can_flush=False)
    eval.plot_mse(
        data0999, 
        label, 
        label=None, 
        ylim=[0, np.max(data0999[len(data0999)//4:])*1.1],
        sub=sub3, 
        can_flush=False)

    sub1.set_title('r=1.0000')
    sub2.set_title('r=0.9999')
    sub3.set_title('r=0.9990')
    sub2.set_ylabel('異常度')
    plt.xlabel('時間ステップ')
    fig.tight_layout()
    plt.show()


# %%
