import numpy as np

class ChangeFinder:
    def __init__(self, d=1, k=1, r=0.5):
        self.mu = np.zeros((d,))
        self.c = np.zeros((k, d, d))
        self.r = r

        self.__k = k
        self.__cur = 0

        self.__xt = np.zeros((k, d))

    def update(self, x)
        def x_t(self, j):
        return self.__x[self.__cur-j]
        # x_t
        self.__xt[self.__cur] = x

        self.mu = (1 - self.r) * self.mu + self.r * x
        for j in range(k):
            self.c[j] = (1-self.r) * self.c[j] + self.r * np.matmul(
                (self.x_t(0) - self.mu),
                (self.x_t(j) - self.mu).T)
        



        self.__cur = (self.__cur + 1) % self.__k 
            