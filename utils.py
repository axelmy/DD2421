import tensorflow as tf
tfk = tf.keras

@tf.function
def preprocess_iv3(x, y):
    return tfk.applications.inception_v3.preprocess_input(x), y

def to_unitrange(x):
    x = x.copy()
    x -= x.min()
    x /= x.max()
    return x