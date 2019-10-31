import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class OS_ELM_Base:
    def __init__(self, in_shape, hidden_shape, out_shape, batch_size, constant, forget_fact=1, reccurent=False, name='', can_normalize=False):
        """
            Input:
                constant : 1 / C
                forget_fact: in (0, 1]
        """
        def iden(shape):
            return tf.diag(tf.ones((shape,)))

        can_normalize = can_normalize
        def norm(y):
            if not can_normalize:
                return y
            mean, variance = tf.nn.moments(y, [1], keep_dims=True)
            return (y - mean) / tf.sqrt(variance + 1e-5)

        t = tf.transpose
        
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
                tf.zeros(shape=[1]+hidden_shape), name='variable_b')

            self.new_w = tf.placeholder(tf.float32, shape=in_shape+hidden_shape, name='placeholder_new_w')
            self.assign_w = tf.assign(self.w, self.new_w)

            mat = self.input @ self.w + tf.tile(self.b, batch_size+[1])

            self.reccurent = reccurent
            if reccurent:
                self.v = tf.Variable( 
                    tf.truncated_normal(                   
                        shape=hidden_shape+hidden_shape, 
                        mean=0.0, 
                        stddev=0.4, # 増幅することを防止する 
                        dtype=tf.float32), name='variable_v')

                self.new_v = tf.placeholder(tf.float32, shape=hidden_shape+hidden_shape, name='placeholder_v')
                self.assign_v = tf.assign(self.v,
                    self.new_v
                )

                self.h = tf.Variable(
                    tf.nn.relu(mat), name='variable_h')
                self.update_h = tf.assign(self.h,
                    tf.nn.relu(mat + self.h @ self.v)
                )
            else:
                self.h = tf.nn.sigmoid(norm(mat))
        
            self.p = tf.Variable(
                tfp.math.pinv(t(self.h) @ self.h), name='variable_p')
            self.beta = tf.Variable(
                    self.p @ t(self.h) @ self.true,
                    name='variable_beta')

            self.output = tf.Variable(tf.zeros(shape=batch_size+out_shape))
            self.update_output = tf.assign(self.output,
                self.h @ self.beta
            )

            f_p = self.p / (forget_fact * forget_fact)
            self.inv = iden(batch_size[0]) + self.h @ f_p @ t(self.h)

            self.update_p = tf.assign(self.p,
                f_p - f_p @ t(self.h) @ tfp.math.pinv(self.inv) @ self.h @ f_p
            )
            
            diff = self.true - self.update_output

            self.update_beta = tf.assign(self.beta,
                self.beta / forget_fact + self.update_p @ t(self.h) @ diff / forget_fact
            )

            self.cost = tf.reduce_mean(tf.square(diff))

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

    def loss(self, x, y, session):
        return session.run(self.cost, feed_dict={
            self.input : x,
            self.true  : y})


class OS_ELM(OS_ELM_Base):
    def __init__(self, in_shape, hidden_shape, out_shape, batch_size, constant=0.001, forget_fact=1, name='os_elm', can_normalize=False):
        super().__init__(
            in_shape, 
            hidden_shape, 
            out_shape, 
            batch_size,
            constant, 
            forget_fact, 
            False,
            name,
            can_normalize)


class OS_ELM_Rec(OS_ELM_Base):
    def __init__(self, in_shape, hidden_shape, out_shape, batch_size, constant=0.001, forget_fact=1, name='os_elm_rec', can_normalize=False):
        super().__init__(
            in_shape, 
            hidden_shape, 
            out_shape, 
            batch_size,
            constant, 
            forget_fact, 
            True,
            name,
            can_normalize)
        # self.v_value = np.random.normal(size=hidden_shape+hidden_shape)
        

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
    # def train(self, x, y, session, v=None, can_apply_trans_beta=False):
    #     if v is not None:
    #         self.v_value = v
        
    #     session.run(self.update_h, feed_dict={
    #         self.input : x,
    #         self.v     : self.v_value})

    #     beta = session.run(self.update_beta, feed_dict={
    #         self.true  : y})
        
    #     if can_apply_trans_beta:
    #         session.run(self.update_w, feed_dict={
    #             self.new_w : beta.T})

class OR_ELM:
    def __init__(self, in_shape, hidden_shape, out_shape, batch_size=1, constant=1e-5, forget_fact=1, config=None):
        """
            Input:
                constant : 1 / C
                forget_fact: in (0, 1]
        """
        self.elmrc = OS_ELM_Rec(in_shape, hidden_shape, out_shape, batch_size, constant, forget_fact, can_normalize=True)
        self.elm_ae_iw = OS_ELM(in_shape, hidden_shape, in_shape, batch_size, constant, forget_fact, 'elm_ae_iw', can_normalize=True)
        self.elm_ae_hw = OS_ELM(hidden_shape, hidden_shape, hidden_shape, batch_size, constant, forget_fact, 'elm_ae_hw', can_normalize=True)
        self.output = self.elmrc.output

    def train(self, x, y, session):
        h = session.run(self.elmrc.h)
        self.elm_ae_hw.train(h, h, session, can_apply_trans_beta=True)
        self.elm_ae_iw.train(x, x, session, can_apply_trans_beta=True)

        v = session.run(self.elm_ae_hw.w)
        w = session.run(self.elm_ae_iw.w)
        self.elmrc.update_w(w, session)
        self.elmrc.update_v(v, session)
        self.elmrc.train(x, y, session, can_apply_trans_beta=False)

    def loss(self, x, y, session):
        return self.elmrc.loss(x, y, session)
