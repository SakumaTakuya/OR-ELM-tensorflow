#%%
import numpy as np
import os
import sys
import tensorflow as tf
sys.path.append('/home/sakuma/work/python/OR-ELM-tensorflow/')

import or_elm as elm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#%%
if __name__ == '__main__':
    y = np.load('data/y_pose.npy')[:1000]
    X = np.load('data/x_pose.npy')[:1000]
    
    X_0 = X[y==6]
    X_1 = X[y==5]

    del X
    del y

    # n * T * C
    in_shape = X_0.shape[2]
    print("shape:", X_0.shape)
    model = elm.OR_ELM([in_shape], [in_shape // 4], [in_shape])
    for x in X_0:
        loss = model.loss(x, x)
        print(f"\rloss:{loss}", end="")
        model.train(x, x)
    
