import numpy as np
from sklearn.decomposition import PCA
import tqdm

normal_data_path = [
    '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_0.csv',
    '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_1.csv',
    '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_2.csv',
    '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_3.csv',
    '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_4.csv',
    '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_5.csv',
    '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_6.csv',
    '/home/sakuma/work/python/OR-ELM-tensorflow/data/data_normal_7.csv',
]


def auto_pca(X, ratio=0.99):
    pca = None
    for i in range(5, 455):
        pca = PCA(n_components=i)
        pca.fit(X)
        if np.sum(pca.explained_variance_ratio_) > 0.99:
            break

    return np.matmul(pca.components_, X.T).T

def create_subseq(ts, look_back, pred_length):
    """
        return:
            sub_seq: len(sub_seq) = len(ts) - look_back - pred_length
                     look_back分のデータを受け取り、pred_length分だけ予測するためその分だけ遅れて出力する
    """
    sub_seq, next_values = [], []
    for i in range(len(ts)-look_back-pred_length):  
        sub_seq.append(ts[i:i+look_back])
        next_values.append(ts[i+look_back:i+look_back+pred_length])
    return sub_seq, next_values

def get_normal_data(look_back, pred_length, max_len=1092):
    sub_datas = [] # batch_size * time * look_back * in_shape
    nex_datas = [] # batch_size * time * pred_length * in_shape
    for p in tqdm.tqdm(normal_data_path):
        sub, nex = create_subseq(
            np.loadtxt(p, delimiter=',')[:max_len], # time * channel
            look_back, 
            pred_length)
        sub_datas.append(sub)
        nex_datas.append(nex)

    return np.array(sub_datas), np.array(nex_datas)
