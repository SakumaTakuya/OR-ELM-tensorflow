#%%
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

#%%
x = np.linspace(0.2, 1, 100)
y = 0.8 * x + np.random.randn(100) * 0.05
X = np.vstack([x, y]).T

np.random.shuffle(X)

plt.scatter(x,y)
plt.show()

#%%
pca = PCA(n_components=1)
pca.fit(X)

# %%
pcaed = np.matmul(pca.components_, X.T)