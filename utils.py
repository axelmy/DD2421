import tensorflow as tf
tfk = tf.keras

@tf.function
def preprocess_iv3(x, *args):
    return (tfk.applications.inception_v3.preprocess_input(x), *args)

def to_unitrange(x):
    x = x.copy()
    x -= x.min()
    x /= x.max()
    return x