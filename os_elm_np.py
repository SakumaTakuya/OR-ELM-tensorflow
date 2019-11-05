import numpy as np


relu = lambda x: np.maximum(0, x)

class OS_ELM:
    def __init__(self, hi_shape, init_x, init_y, act=relu, forget_fact=1):
        assert len(init_x) == len(init_y)
        assert len(init_x) > hi_shape # 優決定性でないと解けない

        in_shape = init_x.shape[1]

        self.forget_fact = forget_fact

        self.hi_shape = hi_shape
        self.act = act

        self.w = np.random.normal(size=[in_shape, hi_shape])
        self.b = np.random.normal(size=[hi_shape])
        self.h = act(init_x @ self.w + self.b)
        # 列フルランクであることを仮定している　→　各軸が独立でないといけない
        self.p = np.linalg.inv(self.h.T @ self.h)
        self.beta = self.p @ self.h.T @ init_y
        self.output = self.h @ self.beta

    def train(self, x, y):
        f_p = self.p / (self.forget_fact * self.forget_fact)

        self.h = self.act(x @ self.w + self.b)
        self.output = self.h @ self.beta
        inv = np.eye(len(x)) + self.h @ f_p @ self.h.T
        self.p = f_p - f_p @ self.h.T @ np.linalg.inv(inv) @ self.h @ f_p
        self.beta = self.beta / self.forget_fact + self.p @ self.h.T @ (y - self.output) / self.forget_fact
    

class Pinv_OS_ELM:
    """
        劣決定性、優決定性でも問題なく使用できる
        Grevilleの方法により疑似逆上列を計算する
    """
    def __init__(self, hi_shape, init_x, init_y, act=relu, forget_fact=1):
        in_shape = init_x.shape[1]

        self.forget_fact = forget_fact

        self.hi_shape = hi_shape
        self.act = act

        self.w = np.random.normal(size=[in_shape, hi_shape])
        self.b = np.random.normal(size=[hi_shape])
        self.h = self.__get_h(init_x)

        # 劣決定性 → ノルム最小の解、　優決定性 → 最小二乗法の解析解
        self.beta = np.linalg.pinv(self.h) @ init_y

        prev_pH = np.linalg.pinv(self.h[:-1]) 
        
        k = self.h[-1:] @ prev_pH
        assert k.shape == (1, init_y.shape[0]-1)
        j = self.h[-1:] - k @ self.h[:-1]
        assert j.shape == (1, hi_shape)
        self.i = (j / j @ j.T).T if np.all(j != 0) else ((k @ prev_pH.T) / (1 + k @ k.T)).T
                
    def output(self):
        return self.h @ self.beta

    def __get_h(self, x):
        return self.act(x @ self.w + self.b)

    def train(self, x, y):
        assert len(x) == len(y) == 1
        # assert len(y) == 1

        self.h = self.__get_h(x)

        tmp = y - self.h @ self.beta
        assert tmp.shape == (1, y.shape[1])

        nex_beta = self.beta + self.i @ tmp
        tmp_dot = tmp @ tmp.T

        # 全ての成分が0の場合は差分がないので更新する必要がない
        if np.all(tmp_dot != 0):
            self.i = (nex_beta - self.beta) @ tmp.T / tmp_dot

        self.beta = nex_beta


class All_Pinv_OS_ELM:
    def __init__(self, hi_shape, init_x, init_y, act=relu, forget_fact=1):
        in_shape = init_x.shape[1]

        self.forget_fact = forget_fact

        self.hi_shape = hi_shape
        self.act = act

        self.w = np.random.normal(size=[in_shape, hi_shape])
        self.b = np.random.normal(size=[hi_shape])

        self.h = self.__get_h(init_x)

        self.H = self.h
        self.pH = np.linalg.pinv(self.H)
        self.Y = init_y
        self.beta = self.pH @ self.Y


    def output(self):
        return self.h @ self.beta

    def __get_h(self, x):
        return self.act(x @ self.w + self.b)

    def next(self, h):
        k = h @ self.pH
        j = h - k @ self.H
        i = j / (j @ j.T) if np.all(j != 0) else k @ self.pH.T / (1 + k @ k.T)

        self.H = np.concatenate((self.H, h))
        self.pH = np.concatenate([self.pH - i.T @ k, i.T], axis=1)

    def train(self, x, y):
        self.h = self.__get_h(x)
        self.next(self.h)

        self.Y = np.concatenate((self.Y, y))
        self.beta = self.pH @ self.Y

if __name__ == '__main__':
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

    data_no = '0'
    look_back = 3
    pred_length = 1

    in_shape = 455
    max_length = 1092 * 8

    normal_data = np.loadtxt(f'/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_{data_no}.csv', delimiter=',')[:max_length//4]
    anormaly_data = np.loadtxt(f'/home/sakuma/work/python/OR-ELM-tensorflow/data/data_{data_no}.csv', delimiter=',')[:max_length]

    normal_label = np.zeros((len(normal_data)))
    anormaly_label = np.loadtxt(f'data/label_{data_no}.csv', delimiter=',')[:max_length]
    
    normal_data, normal_nex = preprocessing.create_subseq(normal_data, look_back, pred_length)
    anormaly_data, anormaly_nex = preprocessing.create_subseq(anormaly_data, look_back, pred_length)

    normal_data = np.array(normal_data)
    normal_nex = np.array(normal_nex)
    anormaly_data = np.array(anormaly_data)
    anormaly_nex = np.array(anormaly_nex)

    def plot(anorm_pred, path_base):
        path = lambda base: os.path.join(
                '.',
                'img', 
                path_base, 
                base, 
                f'no{data_no}_look{look_back:03}_pred{pred_length:03}.png')

        anorm_mse = eval.calcu_mse(anormaly_nex, anorm_pred)
        eval.plot_mse(anorm_mse, anormaly_label, cut=look_back, save_path=path('anorm_mse'))
        eval.plot_auc(anorm_mse, anormaly_label[-len(anorm_mse):], save_path=path('auc'))

    elm = All_Pinv_OS_ELM(
        hi_shape=in_shape // 4 * pred_length,
        forget_fact=0.99,
        init_x=normal_data.reshape((-1, in_shape * look_back)),
        init_y=normal_nex.reshape((-1, in_shape * pred_length)))

    preds = []
    for x, y in tqdm.tqdm(zip(anormaly_data, anormaly_nex)):
    # for x, y in zip(anormaly_data, anormaly_nex):
        x = x.reshape((1, in_shape * look_back))
        y = y.reshape((1, in_shape * pred_length))
        elm.train(x, y)
        out = elm.output()
        assert not np.any(np.isnan(out))
        preds.append(out.reshape(pred_length, in_shape))

    preds = np.array(preds)
    plot(preds, '')



    


