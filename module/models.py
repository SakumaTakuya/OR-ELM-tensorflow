import tensorflow as tf


def get_simple_rnn(
        in_shape, 
        hidden_shape, 
        out_shape, 
        time_step=3,
        pred_step=1,
        loss='mean_squared_error',
        optimizer='sgd',
        layer=tf.keras.layers.GRU):
        """
        参考:
        "Long short term memory networks for anomaly detection in time series.",Malhotra, Pankaj, et al, 2015.
        https://www.renom.jp/ja/notebooks/tutorial/time_series/lstm-anomalydetection/notebook.html
        """
        inputs = tf.keras.Input((None, time_step, in_shape))
        flat = tf.keras.layers.Reshape((-1, time_step * in_shape))(inputs)
        hidden = layer(
            hidden_shape * (time_step + pred_step) // 2,
            return_sequences=True, # return the last output in the output sequence, or the full sequence
            )(flat)
        dense = tf.keras.layers.Dense(pred_step * out_shape)(hidden)
        output = tf.keras.layers.Reshape((-1, pred_step, out_shape))(dense)

        model = tf.keras.models.Model(inputs=inputs, outputs=output)
        model.compile(
            loss=loss, 
            optimizer=optimizer,
            metrics=['mse'])

        return model

