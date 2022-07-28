import tensorflow as tf 


def save_callback():
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir = "logs/", histogram_freq = 1)
    return tb_callback