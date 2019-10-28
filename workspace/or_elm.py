import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class OS_ELM_Base:
    def __init__(self, in_shape, hidden_shape, out_shape, batch_size, constant, forget_fact=1, reccurent=False, name=''):
        """
            Input:
                constant : 1 / C
                forget_fact: in (0, 1]
        """
        def iden(shape):
            return tf.diag(tf.ones((shape,)))

        def norm(y):
            mean, variance = tf.nn.moments(y, [1], keep_dims=True)
            return (y - mean) / tf.sqrt(variance + 1e-5)
        
        with tf.name_scope(name):
            in_shape = list(in_shape)
            hidden_shape = list(hidden_shape)
            out_shape = list(out_shape)
            batch_size = [batch_size]

            self.input = tf.placeholder(tf.float32, shape=batch_size+in_shape, name='placeholder_input')
            self.true = tf.placeholder(tf.float32, shape=batch_size+out_shape, name='placeholder_output')

            self.w = tf.Variable(
                tf.truncated_normal(
                    shape=in_shape+hidden_shape, 
                    mean=0.0, 
                    stddev=1.0, 
                    dtype=tf.float32), name='variable_w')
            self.b = tf.Variable(
                tf.truncated_normal(
                    shape=[1]+hidden_shape, 
                    mean=0.0, 
                    stddev=1.0, 
                    dtype=tf.float32), name='variable_b')

            self.new_w = tf.placeholder(tf.float32, shape=in_shape+hidden_shape, name='placeholder_new_w')
            self.update_w = tf.assign(self.w, self.new_w)

            mat = tf.matmul(self.input, self.w) + tf.tile(self.b, batch_size+[1])

            self.reccurent = reccurent
            if reccurent:
                self.v = tf.placeholder(tf.float32, shape=hidden_shape+hidden_shape, name='placeholder_v')

                self.h = tf.Variable(tf.zeros(shape=batch_size+hidden_shape), name='variable_h')
                self.update_h = tf.assign(self.h,
                    tf.nn.relu(norm(mat + tf.matmul(self.h, self.v)))
                )
            else:
                self.h = tf.nn.relu(norm(mat))

            self.p = tf.Variable(constant * iden(hidden_shape[0]), name='variable_p')
            self.beta = tf.Variable(tf.zeros(shape=hidden_shape+out_shape), name='variable_beta')

            self.output = tf.matmul(self.h, self.beta)

            self.inv = tfp.math.pinv(
                            forget_fact * forget_fact * iden(batch_size[0]) + (
                            forget_fact * tf.matmul(
                                tf.matmul(
                                    self.h,
                                    self.p),
                                self.h, transpose_b=True)))

            self.update_p = tf.assign(self.p,
                1 / forget_fact * self.p - tf.matmul(
                    tf.matmul(
                        self.p,
                        self.h, transpose_b=True),
                    tf.matmul(
                        self.inv,
                        tf.matmul(
                            self.h,
                            self.p)))
            )

            diff = self.true - self.output

            self.update_beta = tf.assign(self.beta,
                self.beta + tf.matmul(
                    tf.matmul(self.update_p, self.h, transpose_b=True),
                    diff)
            )

            self.cost = tf.reduce_mean(tf.square(diff))

    def train(self, x, y, session, can_apply_trans_beta=False):
        beta = session.run(self.update_beta, feed_dict={
            self.input : x,
            self.true  : y})
        
        if can_apply_trans_beta:
            session.run(self.update_w, feed_dict={
                self.new_w : beta.T})

    def assign_weight(self, w, session):
        session.run(self.update_w, feed_dict={
            self.new_w : w})

    def loss(self, x, y, session):
        return session.run(self.cost, feed_dict={
            self.input : x,
            self.true  : y})


class OS_ELM(OS_ELM_Base):
    def __init__(self, in_shape, hidden_shape, out_shape, batch_size, constant, forget_fact=1, name='os_elm'):
        super().__init__(
            in_shape, 
            hidden_shape, 
            out_shape, 
            batch_size,
            constant, 
            forget_fact, 
            False,
            name)


class OS_ELM_Rec(OS_ELM_Base):
    def __init__(self, in_shape, hidden_shape, out_shape, batch_size, constant, forget_fact=1, name='os_elm_rec'):
        super().__init__(
            in_shape, 
            hidden_shape, 
            out_shape, 
            batch_size,
            constant, 
            forget_fact, 
            True,
            name)

    def update_hidden(self, v, session):
        session.run(self.v, feed_dict={
            self.v : v})

    def train(self, x, y, session, v=None, can_apply_trans_beta=False):
        if v is None:
            super().train(x, y, session, can_apply_trans_beta)
        else:
            session.run(self.update_h, feed_dict={
                self.input : x,
                self.v     : v})

            beta = session.run(self.update_beta, feed_dict={
                self.true  : y})
            
            if can_apply_trans_beta:
                session.run(self.update_w, feed_dict={
                    self.new_w : beta.T})

class OR_ELM:
    def __init__(self, in_shape, hidden_shape, out_shape, batch_size=1, constant=1e-5, forget_fact=1, config=None):
        """
            Input:
                constant : 1 / C
                forget_fact: in (0, 1]
        """
        self.elmrc = OS_ELM_Rec(in_shape, hidden_shape, out_shape, batch_size, constant, forget_fact)
        self.elm_ae_iw = OS_ELM(in_shape, hidden_shape, in_shape, batch_size, constant, forget_fact, 'elm_ae_iw')
        self.elm_ae_hw = OS_ELM(hidden_shape, hidden_shape, hidden_shape, batch_size, constant, forget_fact, 'elm_ae_hw')

        init_op = tf.global_variables_initializer()
        self.session = tf.Session(config=config)
        self.session.run(init_op)


    def train(self, x, y):
        h = self.session.run(self.elmrc.h)
        self.elm_ae_hw.train(h, h, self.session, can_apply_trans_beta=True)
        self.elm_ae_iw.train(x, x, self.session, can_apply_trans_beta=True)

        v = self.session.run(self.elm_ae_hw.w)
        w = self.session.run(self.elm_ae_iw.w)
        self.elmrc.assign_weight(w, self.session)
        self.elmrc.train(x, y, self.session, v, can_apply_trans_beta=False)

    def loss(self, x, y):
        return self.elmrc.loss(x, y, self.session)

    def close(self):
        self.session.close()
