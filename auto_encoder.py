#%%
import changefinder 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
from sklearn import metrics
sys.path.append('/home/sakuma/work/python/OR-ELM-tensorflow/')

import or_elm as elm

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
# T * C
in_shape = X.shape[1]
print("shape:", X.shape)
model = elm.OR_ELM([in_shape], [in_shape // 4], [in_shape], forget_fact=1)

cfs = [changefinder.ChangeFinder() for i in range(in_shape)]
for x in nX:
    map(lambda cf, x_n: cf.update(x_n), cfs, x)
    x = x[np.newaxis,:]
    model.train(x, x)

losses = []
scores = []
for x in X:
    score = sum([cf.update(x_n) for cf, x_n in zip(cfs, x)]) 
    x = x[np.newaxis,:]
    loss = model.loss(x, x)
    print(f"\ror_elm loss:{loss} | cf score:{score}", end="")
    model.train(x, x)

    losses.append(loss)
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
