#%%
import numpy as np

#%%
def generate_variational_weight(shape):
    return (np.random.randint(0, 2, shape) * 2 - 1) * 0.1


def generate_reservoir_weight(shape, forget_fact=0.999):
    w = np.random.normal(0, 1, shape)
    sr = max(abs(np.linalg.eigvals(w)))
    return w / sr * forget_fact


class RC_OS_ELM:
    def __init__(self, 
        hi_shape, 
        init_x, 
        init_y, 
        act=np.tanh, 
        forget_fact=1, 
        leak_rate=0.9, 
        regularizer=1/np.e**5,
        max_diff=1.5,
        gen_w_in=generate_variational_weight,
        gen_b=generate_variational_weight,
        gen_w_res=generate_reservoir_weight):

        assert len(init_x) == len(init_y)

        in_shape = init_x.shape[1]

        self.forget_fact = forget_fact
        self.leak_rate = leak_rate

        self.hi_shape = hi_shape
        self.act = act

        self.w_in = gen_w_in([in_shape, hi_shape])
        self.w_res = gen_w_res([hi_shape, hi_shape])
        self.b = gen_b([hi_shape])

        self.h = act(leak_rate * (init_x @ self.w_in + self.b))
        self.p = np.linalg.inv(np.random.rand(hi_shape * hi_shape).reshape((hi_shape, hi_shape)) * 0.01)#self.h.T @ self.h + regularizer * np.eye(hi_shape))
        self.beta = self.p @ self.h.T @ init_y

        self.out = self.h @ self.beta
        self.loss = 0
        self.inv = 0

    def output(self):
        return self.out

    def input(self, x):
        self.h = self.act((1 - self.leak_rate) * self.h + self.leak_rate * (x @ self.w_in + self.b + self.h @ self.w_res))
        self.out = self.h @ self.beta

    def update(self, h, y):
        f_p = self.p / (self.forget_fact * self.forget_fact)
        self.inv = np.linalg.inv(np.eye(len(y)) + h @ f_p @ h.T)

        self.p = f_p - f_p @ h.T @ self.inv @ h @ f_p
        self.beta = self.beta + self.p @ h.T @ (y - self.out)

        # # power iteration によって最大特異値を推定
        # v = self.beta.T @ self.u
        # u = self.beta.T @ self.v

        # self.v = v / np.linalg.norm(v, ord=2)
        # self.v = u / np.linalg.norm(u, ord=2)
        
        # if np.abs(self.u.T @ self.beta @ self.v) > 1:
        #     self.inv = np.linalg.inv(np.eye(len(y)) + h @ f_p @ h.T)

        #     self.p = f_p - f_p @ h.T @ self.inv @ h @ f_p
        #     self.beta = self.beta + self.p @ h.T @ (y - self.out)

    def train(self, x, y):
        self.input(x)
        self.update(self.h, y)

class RC_Max_Clip_OS_ELM:
    def __init__(self, 
        hi_shape, 
        init_x, 
        init_y, 
        act=np.tanh, 
        forget_fact=1, 
        leak_rate=0.9, 
        regularizer=1/np.e**5,
        max_diff=1.5,
        gen_w_in=generate_variational_weight,
        gen_b=generate_variational_weight,
        gen_w_res=generate_reservoir_weight):

        assert len(init_x) == len(init_y)

        in_shape = init_x.shape[1]

        self.forget_fact = forget_fact
        self.leak_rate = leak_rate

        self.hi_shape = hi_shape
        self.act = act

        self.w_in = gen_w_in([in_shape, hi_shape])
        self.w_res = gen_w_res([hi_shape, hi_shape])
        self.b = gen_b([hi_shape])

        self.h = act(leak_rate * (init_x @ self.w_in + self.b))
        self.p = np.linalg.inv(self.h.T @ self.h + regularizer * np.eye(hi_shape))
        self.beta = self.p @ self.h.T @ init_y

        self.out = self.h @ self.beta
        self.loss = 0
        self.inv = 0

        # self.u = np.random.normal(0, 1, [self.beta.shape[0]])
        # self.v = np.random.normal(0, 1, [self.beta.shape[1]])
        self.max_diff = max_diff

    def output(self):
        return self.out

    def input(self, x):
        self.h = self.act((1 - self.leak_rate) * self.h + self.leak_rate * (x @ self.w_in + self.b + self.h @ self.w_res))
        self.out = self.h @ self.beta

    def update(self, h, y):
        f_p = self.p / (self.forget_fact * self.forget_fact)
        self.inv = np.linalg.inv(np.eye(len(y)) + h @ f_p @ h.T)

        p = f_p - f_p @ h.T @ self.inv @ h @ f_p
        diff = np.linalg.norm(self.p - p, ord=2)
        if diff > self.max_diff:
            # self.max_diff = diff
            f_p = self.p
            self.inv = np.linalg.inv(np.eye(len(y)) + h @ f_p @ h.T)
        
        self.p = f_p - f_p @ h.T @ self.inv @ h @ f_p
        self.beta = self.beta + self.p @ h.T @ (y - self.out)

    def train(self, x, y):
        self.input(x)
        self.update(self.h, y)


class RC_Delay_OS_ELM(RC_OS_ELM):
    def __init__(self, 
        look_back,
        pred_len,
        hi_shape,
        delay,
        init_x, 
        init_y, 
        outlier_rate= 0.999,
        act=np.tanh, 
        forget_fact=1, 
        leak_rate=0.9, 
        regularizer=1/np.e**5,
        gen_w_in=generate_variational_weight,
        gen_b=generate_variational_weight,
        gen_w_res=generate_reservoir_weight):
        super().__init__(
            hi_shape,
            init_x,
            init_y,
            act,
            forget_fact,
            leak_rate,
            regularizer,
            gen_w_in,
            gen_b,
            gen_w_res)
        self.out_queue_len = look_back // pred_len
        self.delay = delay
        self.pred_len = pred_len
        self.outlier_rate = outlier_rate
        self.loss_queue = []
        self.output_queue = []

    def train(self, x, y):
        prev_h = self.h
        self.input(x)

        loss = np.sum(np.square(y - self.out))
        
        self.output_queue.append(self.out)

        del self.loss_queue[:-self.delay]
        del self.output_queue[:-self.out_queue_len]
        
        # 毎度delay個前のデータのロスを見て学習するか判断する
        if  len(self.loss_queue) == self.delay and \
            len(self.output_queue) == self.out_queue_len and \
            loss > np.max(self.loss_queue) * self.outlier_rate:

            self.h = prev_h
            self.input(np.array(self.output_queue).reshape(x.shape))

        self.loss_queue.append(loss)
        self.update(self.h, y)


class RC_FP_ELM:
    def __init__(self, 
        hi_shape, 
        init_x, 
        init_y, 
        act=np.tanh, 
        forget_fact=1, 
        leak_rate=0.9, 
        regularizer=1/np.e**5,
        gen_w_in=generate_variational_weight,
        gen_b=generate_variational_weight,
        gen_w_res=generate_reservoir_weight):

        assert len(init_x) == len(init_y)

        in_shape = init_x.shape[1]

        self.forget_fact = forget_fact
        self.leak_rate = leak_rate
        self.regularizer = regularizer

        self.hi_shape = hi_shape
        self.act = act

        self.w_in = gen_w_in([in_shape, hi_shape])
        self.w_res = gen_w_res([hi_shape, hi_shape])
        self.b = gen_b([hi_shape])

        self.h = act(leak_rate * (init_x @ self.w_in + self.b))
        self.k = self.h.T @ self.h
        self.beta = (regularizer * np.eye(len(self.k)) + self.k) @ self.h.T @ init_y
        self.out = self.h @ self.beta

    def output(self):
        return self.out

    def input(self, x):
        self.h = self.act((1 - self.leak_rate) * self.h + self.leak_rate * (x @ self.w_in + self.b + self.h @ self.w_res))
        self.out = self.h @ self.beta

    def update(self, h, y):
        self.k = self.forget_fact ** 2 * self.k + h.T @ h
        inv = np.linalg.inv(self.regularizer * np.eye(len(self.k)) + self.k) 
        self.beta = self.beta + inv @ (h.T @ (y - h @ self.beta) - self.regularizer * (1 - self.forget_fact ** 2) * self.beta)

    def train(self, x, y):
        self.input(x)
        self.update(self.h, y)



# 遅延付き学習をどのように行うか
# 仮Reservoirを用意しておいてN個分は仮Reservoirを使用する
# N個たまったらロスの大きなものは入力を予測値に置き換える

# 現状正則化項が最初にしかかからないことになっているが…→RC_FP_ELMのほうが精度が良ければ考える→なんかよくないので無視