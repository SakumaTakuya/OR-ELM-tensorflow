#%% import
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
import sys
import tensorflow as tf
import tqdm 

sys.path.append('/home/sakuma/work/python/OR-ELM-tensorflow/')

from module import eval
from module import or_elm
from module import preprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

type_choice = ['elm_rec', 'elm', 'gru', 'or_elm']
data_choice = ['0', '1', '2', '3', '4', '5', '6', '7']

parser = argparse.ArgumentParser()
parser.add_argument('--type', choices=type_choice, default=type_choice[0])
parser.add_argument('--data', choices=data_choice, default=data_choice[0])
parser.add_argument('--look', type=int, default=3)
parser.add_argument('--pred', type=int, default=1)
parser.add_argument('--pref', type=str, default='')

#%% main
if __name__ == '__main__':
    args = parser.parse_args()
    pred_type = args.type
    data_no = args.data
    look_back = args.look
    pred_length = args.pred
    prefix = args.pref

#%% define params
    in_shape = 455
    max_length = 1092 * 8

#%% load base data
    # 0番はgruで学習に使用している
    normal_data = np.loadtxt(f'data/data_normal_{data_no}.csv', delimiter=',')[:max_length//4]
    anormaly_data = np.loadtxt(f'data/data_{data_no}.csv', delimiter=',')[:max_length]

    normal_label = np.zeros((len(normal_data)))
    anormaly_label = np.loadtxt(f'data/label_{data_no}.csv', delimiter=',')[:max_length]
#%% create data
    normal_data, normal_nex = preprocessing.create_subseq(normal_data, look_back, pred_length)
    anormaly_data, anormaly_nex = preprocessing.create_subseq(anormaly_data, look_back, pred_length)

    normal_data = np.array(normal_data)
    normal_nex = np.array(normal_nex)
    anormaly_data = np.array(anormaly_data)
    anormaly_nex = np.array(anormaly_nex)

#%% define utils
    def plot(norm_pred, anorm_pred, path_base):
        path = lambda base: os.path.join(
                '.',
                'img', 
                path_base, 
                base, 
                f'no{data_no}_look{look_back:03}_pred{pred_length:03}{prefix}.png')

        norm_mse = eval.calcu_mse(normal_nex, norm_pred)
        eval.plot_mse(norm_mse, normal_label, cut=look_back, save_path=path('norm_mse'))

        anorm_mse = eval.calcu_mse(anormaly_nex, anorm_pred)
        eval.plot_mse(anorm_mse, anormaly_label, cut=look_back, save_path=path('anorm_mse'))
        eval.plot_auc(anorm_mse, anormaly_label[-len(anorm_mse):], save_path=path('auc'))

#%% [os elm] define model
    if 'elm' in pred_type:
        if 'rec' in pred_type:
            elm = or_elm.OS_ELM_Rec(
                [in_shape * look_back],
                [in_shape // 4 * pred_length],
                [in_shape * pred_length],
                1,
                forget_fact=0.9,
                can_normalize=False)
        elif 'or_elm' in pred_type:
            elm = or_elm.OR_ELM(
                [in_shape * look_back],
                [in_shape // 4 * pred_length],
                [in_shape * pred_length],
                forget_fact=0.9)
        else:
            elm = or_elm.OS_ELM(
                [in_shape * look_back],
                [in_shape // 4 * pred_length],
                [in_shape * pred_length],
                1,
                forget_fact=0.5)

#%% [os elm] train and predict
        init_op = tf.global_variables_initializer()
        elm_norm_pred = []
        elm_anorm_pred = []

        with tf.Session() as sess:
            sess.run(init_op, feed_dict={
                elm.input : normal_data,
                elm.train : normal_nex
            })
            # if pred_type == 'elm_rec_dont_learn':
            #     for x, y in tqdm.tqdm(zip(normal_data, normal_nex)):
            #         x = x.reshape((1, in_shape * look_back))
            #         y = y.reshape((1, in_shape * pred_length))
            #         sess.run(elm.update_h, feed_dict={
            #             elm.input : x,
            #             elm.v     : elm.v_value})
            #         out = sess.run(elm.update_output)
            #         elm_norm_pred.append(out.reshape(pred_length, in_shape))
        
            #     for x, y in tqdm.tqdm(zip(anormaly_data, anormaly_nex)):
            #         x = x.reshape((1, in_shape * look_back))
            #         y = y.reshape((1, in_shape * pred_length))
            #         sess.run(elm.update_h, feed_dict={
            #             elm.input : x,
            #             elm.v     : elm.v_value})
            #         out = sess.run(elm.update_output)
            #         elm_anorm_pred.append(out.reshape(pred_length, in_shape))
            # else:
            # for x, y in tqdm.tqdm(zip(normal_data, normal_nex)):
            #     x = x.reshape((1, in_shape * look_back))
            #     y = y.reshape((1, in_shape * pred_length))
            #     elm.train(x, y, sess)
            #     out = sess.run(elm.output)
            #     assert not np.any(np.isnan(out))
            #     elm_norm_pred.append(out.reshape(pred_length, in_shape))
    
            for x, y in tqdm.tqdm(zip(anormaly_data, anormaly_nex)):
                x = x.reshape((1, in_shape * look_back))
                y = y.reshape((1, in_shape * pred_length))
                elm.train(x, y, sess)
                out = sess.run(elm.output)
                assert not np.any(np.isnan(out))
                elm_anorm_pred.append(out.reshape(pred_length, in_shape))

# %% [os elm] evaluation
        elm_norm_pred = np.array(elm_norm_pred)
        elm_anorm_pred = np.array(elm_anorm_pred)
        plot(elm_norm_pred, elm_anorm_pred, path_base=pred_type)

#%% [gru] load model
    elif pred_type == 'gru':
        gru = tf.keras.models.load_model('/home/sakuma/work/simple_gru.h5')

#%% [gru] predict
        normal_data_4_gru = np.array(normal_data)[np.newaxis]
        anormaly_data_4_gru = np.array(anormaly_data)[np.newaxis]

        gru_norm_pred = gru.predict(normal_data_4_gru)
        gru_anorm_pred = gru.predict(anormaly_data_4_gru)

#%% [gru] evaluation
        plot(gru_anorm_pred[0], gru_anorm_pred[0], path_base=pred_type)
    
