#%%
import changefinder 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import metrics
from sklearn.decomposition import PCA
sys.path.append('/home/sakuma/work/python/OR-ELM-tensorflow/')

import or_elm as elm
from rnn import Rnn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#%%
"""
    この学習方法では則適応する
    機構的にオートエンコーダにするのは正しいのか…？
        - 現在のデータと過去のデータから現在のデータを復元ってちょっとおかしい気がする
        - 次の出力を予測するほうがRNNらしい気が、、、？


    ★そもそも異常検知をどのように評価するのか知る必要がありそう
"""
X = np.load('data/x_pose_BackwardWalk.npy')
X_f = np.load('data/x_pose_Walking.npy')

# n * T * C
in_shape = X.shape[2]
print("shape:", X.shape)
model = elm.OR_ELM([in_shape], [in_shape // 4], [in_shape], forget_fact=1)
i = 0
for xfs in X_f:
    for xs in X:
        for x in xs:
            x = x[np.newaxis,:]
            loss = model.loss(x, x)
            print(f"\rbackward loss:{loss}", end="")
            model.train(x, x)

        i+= 1
        if i % 1000 == 0:
            print("")
            for xf in xfs:
                xf = xf[np.newaxis,:]
                loss = model.loss(xf, xf)
                print(f"\rforward loss:{loss}", end="")
                model.train(xf, xf)

            print("")
model.close()

#%%
"""
    次の値を予測するモデル
    AUCで評価をとりたい
    その場で異常を判断していいのか問題
        - 
"""
X = np.load('data/x_pose_BackwardWalk.npy')
X_f = np.load('data/x_pose_Walking.npy')

def data(xs):
    for x in xs:
        yield x[np.newaxis,:]

# n * T * C
in_shape = X.shape[2]
print("shape:", X.shape)
model = elm.OR_ELM([in_shape], [in_shape // 4], [in_shape], forget_fact=1)
i = 0
for xfs in X_f:
    for xs in X:
        gen = data(xs)
        nex = None
        for x in gen:
            if nex is not None: 
                loss = model.loss(nex, x)
                print(f"\rbackward loss:{loss}", end="")
            
            nex = next(gen, None)
            # 次の値を予測するように学習する
            if nex is not None:
                model.train(x, nex)

        i+= 1
        if i % 1000 == 0:
            print("")
            gen = data(xfs)
            nex = None
            for xf in gen:
                if nex is not None:
                    loss = model.loss(xf, xf)
                    print(f"\rforward loss:{loss}", end="")
                
                nex = next(gen, None)
                if nex is not None:
                    model.train(xf, xf)

            print("")
model.close()

#%% change finder
"""
データが正規分布に従うか判定する必要がある
＝ChangeFinderの仮定について正規分布が成り立つか確認

EOF解析によって時系列データの主成分分析を行うことができるか
"""

#%%
nX = np.loadtxt('/home/sakuma/work/python/OR-ELM-tensorflow/data/negative_data_0.csv', delimiter=',')
X = np.loadtxt('/home/sakuma/work/python/OR-ELM-tensorflow/data/data_np_test_0.csv', delimiter=',')
y = np.loadtxt('/home/sakuma/work/python/OR-ELM-tensorflow/data/label_np_test_0.csv', delimiter=',')

#%%
pca = None
for i in range(5, 455):
    pca = PCA(n_components=i)
    pca.fit(X)
    if np.sum(pca.explained_variance_ratio_) > 0.99:
        break

#%%
nX = np.matmul(pca.components_, nX.T).T
X = np.matmul(pca.components_, X.T).T
#%%
# T * C
in_shape = X.shape[1]
print("shape:", X.shape)
model = elm.OR_ELM([in_shape], [in_shape // 2], [in_shape], forget_fact=1)
oselm_r = elm.OS_ELM_Rec(
    [in_shape], 
    [in_shape // 2], 
    [in_shape], 
    batch_size=1,
    constant=1e-5,
    forget_fact=1)
sess = model.session

cfs = [changefinder.ChangeFinder() for i in range(in_shape)]
for x in nX:
    map(lambda cf, x_n: cf.update(x_n), cfs, x)
    x = x[np.newaxis,:]
    oselm_r.train(x, x, sess)
    model.train(x, x)

#%%
losses = []
r_losses = []
scores = []
for x in X:
    score = max([cf.update(x_n) for cf, x_n in zip(cfs, x)]) 
    x = x[np.newaxis,:]
    loss = model.loss(x, x)
    r_loss = oselm_r.loss(x, x, sess)
    print(f"\ror_elm:{loss} | cf:{score} | rec:{r_loss}", end="")
    model.train(x, x)
    oselm_r.train(x, x, sess)

    losses.append(loss)
    r_losses.append(r_loss)
    scores.append(score)

model.close()

#%%
"""
これを見るとChangeFinderでは見つけることができない
"""

# FPR, TPR(, しきい値) を算出
fpr, tpr, thresholds = metrics.roc_curve(np.where(y == 0, 0, 1), losses)

# ついでにAUCも
auc = metrics.auc(fpr, tpr)

# ROC曲線をプロット
plt.plot(fpr, tpr, label='ORELM (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

# FPR, TPR(, しきい値) を算出
fpr, tpr, thresholds = metrics.roc_curve(np.where(y == 0, 0, 1), scores)

# ついでにAUCも
auc = metrics.auc(fpr, tpr)

# ROC曲線をプロット
plt.plot(fpr, tpr, label='Change Finder (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

plt.show()

# %%
for x in X:
    score = max([cf.update(x_n) for cf, x_n in zip(cfs, x)]) 
    x = x[np.newaxis,:]
    loss = model.loss(x, x)
    print(f"\ror_elm loss:{loss} | cf score:{score}", end="")
    model.train(x, x)

    losses.append(loss)
    scores.append(score)

#%%
"""
change finder は一次元向けのモデルであるため、考慮しなくてよさそう
また、次元数が比較的大きいので古典的手法ではなくNN同士で比較する

案
    - RNN
    - STFTスペクトルに変換して入力
    
"""

rnn_model = Rnn()