import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class OS_ELM_Base:
    def __init__(self, in_shape, hidden_shape, out_shape, forget_fact=1, reccurent=False, name='', can_normalize=False):
        """
            Input:
                forget_fact: in (0, 1]
        """
        can_normalize = can_normalize
        def norm(y):
            if not can_normalize:
                return y
            mean, variance = tf.nn.moments(y, [1], keep_dims=True)
            return (y - mean) / tf.sqrt(variance + 1e-5)

        t = tf.transpose
        iden = lambda shape: tf.diag(tf.ones([shape]))
        
        with tf.name_scope(name):
            in_shape = list(in_shape)
            hidden_shape = list(hidden_shape)
            out_shape = list(out_shape)
            batch_size = [None]

            self.input = tf.placeholder(tf.float32, shape=batch_size+in_shape, name='placeholder_input')
            self.true = tf.placeholder(tf.float32, shape=batch_size+out_shape, name='placeholder_output')

            self.w = tf.Variable(
                tf.truncated_normal(
                    shape=in_shape+hidden_shape, 
                    mean=0.0, 
                    stddev=1.0, 
                    dtype=tf.float32), name='variable_w')
            self.b = tf.Variable(
                tf.zeros(shape=[1]+hidden_shape), name='variable_b')

            self.new_w = tf.placeholder(tf.float32, shape=in_shape+hidden_shape, name='placeholder_new_w')
            self.assign_w = tf.assign(self.w, self.new_w)

            mat = self.input @ self.w + self.b

            self.reccurent = reccurent
            if reccurent:
                self.v = tf.Variable( 
                    tf.truncated_normal(                   
                        shape=hidden_shape+hidden_shape, 
                        mean=0.0, 
                        stddev=0.4, # 増幅することを防止する 
                        dtype=tf.float32), validate_shape=False, name='variable_v')

                self.new_v = tf.placeholder(tf.float32, shape=hidden_shape+hidden_shape, name='placeholder_v')
                self.assign_v = tf.assign(self.v,
                    self.new_v
                )

                self.h = tf.Variable(
                    tf.zeros(shape=[1]+hidden_shape), validate_shape=False, name='variable_h')
                self.update_h = tf.assign(self.h,
                    tf.nn.relu(mat + self.h @ self.v),
                    validate_shape=False
                )
            else:
                self.h = tf.nn.relu(norm(mat))
        
            self.p = tf.Variable(
                tf.zeros(shape=hidden_shape+hidden_shape), 
                name='variable_p')
            self.init_p = tf.assign(self.p,
                tfp.math.pinv(t(self.h) @ self.h)
            )

            self.beta = tf.Variable(
                    tf.zeros(shape=hidden_shape+out_shape), 
                    name='variable_beta')
            self.init_beta = tf.assign(self.beta,
                self.init_p @ t(self.h) @ self.true
            )

            self.output = tf.Variable(
                self.true,
                validate_shape=False, name='output')
            self.update_output = tf.assign(self.output,
                self.h @ self.beta,
                validate_shape=False
            )

            f_p = self.p / (forget_fact * forget_fact)
            self.inv = iden(tf.shape(self.input)[0]) + self.h @ f_p @ t(self.h)

            self.update_p = tf.assign(self.p,
                f_p - f_p @ t(self.h) @ tfp.math.pinv(self.inv) @ self.h @ f_p
            )

            self.update_beta = tf.assign(self.beta,
                self.beta / forget_fact \
                + self.update_p @ t(self.h) @ (self.true - self.update_output) / forget_fact
            )

            self.cost = tf.reduce_mean(tf.square(self.true - self.output))

    def train(self, x, y, session, can_apply_trans_beta=False):
        beta = session.run(self.update_beta, feed_dict={
            self.input : x,
            self.true  : y})
        
        if can_apply_trans_beta:
            session.run(self.update_w, feed_dict={
                self.new_w : beta.T})

    def update_w(self, w, session):
        session.run(self.assign_w, feed_dict={
            self.new_w : w})

    def loss(self, y, session):
        return session.run(self.cost, feed_dict={
            self.true  : y})

class OS_ELM(OS_ELM_Base):
    def __init__(self, in_shape, hidden_shape, out_shape, forget_fact=1, name='os_elm', can_normalize=False):
        super().__init__(
            in_shape, 
            hidden_shape, 
            out_shape, 
            forget_fact, 
            False,
            name,
            can_normalize)

class OS_ELM_Rec(OS_ELM_Base):
    def __init__(self, in_shape, hidden_shape, out_shape, forget_fact=1, name='os_elm_rec', can_normalize=False):
        super().__init__(
            in_shape, 
            hidden_shape, 
            out_shape, 
            forget_fact, 
            True,
            name,
            can_normalize)

    def update_v(self, v, session):
        session.run(self.assign_v, feed_dict={
            self.new_v : v})

    def train(self, x, y, session, can_apply_trans_beta=False):
        session.run(self.update_h, feed_dict={
            self.input : x})

        beta = session.run(self.update_beta, feed_dict={
            self.true  : y})
        
        if can_apply_trans_beta:
            session.run(self.update_w, feed_dict={
                self.new_w : beta.T})


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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

    elm = OS_ELM(
        [in_shape * look_back],
        [in_shape // 4 * pred_length],
        [in_shape * pred_length],
        forget_fact=0.9,
        can_normalize=False)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op, feed_dict={
            elm.input : normal_data.reshape((-1, in_shape * look_back)),
            elm.true : normal_nex.reshape((-1, in_shape * pred_length))
        })
        sess.run(elm.init_beta, feed_dict={
            elm.input : normal_data.reshape((-1, in_shape * look_back)),
            elm.true : normal_nex.reshape((-1, in_shape * pred_length))
        })

        for x, y in tqdm.tqdm(zip(anormaly_data, anormaly_nex)):
            x = x.reshape((1, in_shape * look_back))
            y = y.reshape((1, in_shape * pred_length))
            elm.train(x, y, sess)
            out = sess.run(elm.output)
            assert not np.any(np.isnan(out))