import numpy as np


def generate_variational_weight(shape):
    return (np.random.randint(0, 2, shape) * 2 - 1) * 0.1


def generate_reservoir_weight(shape):
    w = np.random.normal(0, 1, shape)
    sr = max(abs(np.linalg.eigvals(w)))
    return w / sr * 0.999


class RC_OS_ELM:
    def __init__(self, 
        hi_shape, 
        init_x, 
        init_y, 
        act=np.tanh, 
        forget_fact=1, 
        leak_rate=0.9, 
        regularizer=1/np.e**5,
        init_w_in=None,
        init_b=None,
        init_w_res=None):

        assert len(init_x) == len(init_y)

        in_shape = init_x.shape[1]

        self.forget_fact = forget_fact
        self.leak_rate = leak_rate

        self.hi_shape = hi_shape
        self.act = act

        self.w_in = generate_variational_weight([in_shape, hi_shape]) if init_w_in is None else init_w_in
        self.w_res = generate_reservoir_weight([hi_shape, hi_shape]) if init_w_res is None else init_w_res
        self.b = generate_variational_weight([hi_shape]) if init_b is None else init_b

        self.h = act(leak_rate * (init_x @ self.w_in + self.b))
        self.p = np.linalg.inv(self.h.T @ self.h + regularizer * np.eye(hi_shape))
        self.beta = self.p @ self.h.T @ init_y

        self.out = self.h @ self.beta

    def output(self):
        return self.out

    def input(self, x):
        self.h = self.act((1 - self.leak_rate) * self.h + self.leak_rate * (x @ self.w_in + self.b + self.h @ self.w_res))
        self.out = self.h @ self.beta

    def update(self, y):
        f_p = self.p / (self.forget_fact * self.forget_fact)
        inv = np.eye(len(y)) + self.h @ f_p @ self.h.T
        self.p = f_p - f_p @ self.h.T @ np.linalg.inv(inv) @ self.h @ f_p
        self.beta = self.beta + self.p @ self.h.T @ (y - self.out)

    def train(self, x, y):
        self.input(x)
        self.update(y)


class RC_FP_ELM:
    def __init__(self, 
        hi_shape, 
        init_x, 
        init_y, 
        act=np.tanh, 
        forget_fact=1, 
        leak_rate=0.9, 
        regularizer=1/np.e**5,
        init_w_in=None,
        init_b=None,
        init_w_res=None):

        assert len(init_x) == len(init_y)

        in_shape = init_x.shape[1]

        self.forget_fact = forget_fact
        self.leak_rate = leak_rate
        self.regularizer = regularizer

        self.hi_shape = hi_shape
        self.act = act

        self.w_in = generate_variational_weight([in_shape, hi_shape]) if init_w_in is None else init_w_in
        self.w_res = generate_reservoir_weight([hi_shape, hi_shape]) if init_w_res is None else init_w_res
        self.b = generate_variational_weight([hi_shape]) if init_b is None else init_b

        self.h = act(leak_rate * (init_x @ self.w_in + self.b))
        self.k = self.h.T @ self.h
        self.beta = (regularizer * np.eye(len(self.k)) + self.k) @ self.h.T @ init_y
        self.out = self.h @ self.beta

    def output(self):
        return self.out

    def input(self, x):
        self.h = self.act((1 - self.leak_rate) * self.h + self.leak_rate * (x @ self.w_in + self.b + self.h @ self.w_res))
        self.out = self.h @ self.beta

    def update(self, y):
        self.k = self.forget_fact ** 2 * self.k + self.h.T @ self.h
        inv = np.linalg.inv(self.regularizer * np.eye(len(self.k)) + self.k) 
        self.beta = self.beta + inv @ ( self.h.T @ (y - self.h @ self.beta) - self.regularizer * (1 - self.forget_fact ** 2) @ self.beta)

    def train(self, x, y):
        self.input(x)
        self.update(y)



# 遅延付き学習をどのように行うか
# 仮Reservoirを用意しておいてN個分は仮Reservoirを使用する
# N個たまったらロスの大きなものは入力を予測値に置き換える

# 現状正則化項が最初にしかかからないことになっているが…→RC_FP_ELMのほうが精度が良ければ考える