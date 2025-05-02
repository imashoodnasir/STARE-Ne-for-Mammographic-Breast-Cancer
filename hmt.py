# hmt.py - STARE-Net module implementation

import tensorflow as tf
from tensorflow.keras.layers import Dense, Add, Concatenate, ReLU

def cross_scale_attention(Q, K, V, d):
    attention_scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(d, tf.float32))
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)
    return tf.matmul(attention_weights, V)

def build_hmt(F1, F2, F3, d=64):
    # Project features to transformer space
    Q12 = Dense(d)(tf.reshape(F1, [tf.shape(F1)[0], -1, F1.shape[-1]]))
    Q13 = Dense(d)(tf.reshape(F1, [tf.shape(F1)[0], -1, F1.shape[-1]]))
    
    K2 = Dense(d)(tf.reshape(F2, [tf.shape(F2)[0], -1, F2.shape[-1]]))
    V2 = Dense(d)(tf.reshape(F2, [tf.shape(F2)[0], -1, F2.shape[-1]]))
    
    K3 = Dense(d)(tf.reshape(F3, [tf.shape(F3)[0], -1, F3.shape[-1]]))
    V3 = Dense(d)(tf.reshape(F3, [tf.shape(F3)[0], -1, F3.shape[-1]]))

    # Hierarchical Attention
    FHA_2 = cross_scale_attention(Q12, K2, V2, d)
    FHA_3 = cross_scale_attention(Q13, K3, V3, d)

    # Concatenate and project
    concat = Concatenate(axis=-1)([FHA_2, FHA_3])
    projected = Dense(d)(concat)
    
    # Residual connection
    F1_flat = tf.reshape(F1, [tf.shape(F1)[0], -1, F1.shape[-1]])
    U = Add()([projected, F1_flat])

    # Feed Forward
    W1 = Dense(d, activation='relu')(U)
    Z = Add()([Dense(d)(W1), U])
    return Z
