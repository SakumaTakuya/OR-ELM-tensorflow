import numpy as np


relu = lambda x: np.maximum(0, x)

class OS_ELM:
    def __init__(self, 
        hi_shape, 
        init_x, 
        init_y, 
        act=relu, 
        forget_fact=1, 
        regularizer=1/np.e**5,
        init_w = None,
        init_b = None):
        assert len(init_x) == len(init_y)

        in_shape = init_x.shape[1]

        self.forget_fact = forget_fact

        self.hi_shape = hi_shape
        self.act = act

        self.w = np.random.normal(size=[in_shape, hi_shape]) if init_w is None else init_w
        self.b = np.random.normal(size=[hi_shape]) if init_b is None else init_b
        self.h = act(init_x @ self.w + self.b)
        self.p = np.linalg.inv(self.h.T @ self.h + regularizer * np.eye(hi_shape))
        self.beta = self.p @ self.h.T @ init_y
        self.out = self.h @ self.beta

    def output(self):
        return self.out

    def input(self, x):
        self.h = self.act(x @ self.w + self.b)
        self.out = self.h @ self.beta

    def update(self, y):
        f_p = self.p / (self.forget_fact * self.forget_fact)
        inv = np.eye(len(y)) + self.h @ f_p @ self.h.T
        self.p = f_p - f_p @ self.h.T @ np.linalg.inv(inv) @ self.h @ f_p
        self.beta = self.beta + self.p @ self.h.T @ (y - self.out)

    def train(self, x, y):
        self.input(x)
        self.update(y)


# 遅延付き学習をどのように行うべきか？？？