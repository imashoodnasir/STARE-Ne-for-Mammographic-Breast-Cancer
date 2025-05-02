# elt.py - STARE-Net module implementation

import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, Add, Concatenate

def compute_entropy(prob_map):
    entropy = -tf.reduce_sum(prob_map * tf.math.log(prob_map + 1e-8) / tf.math.log(2.0), axis=-1)
    return entropy

def select_entropy_windows(entropy_map, window_size=7, top_k=5):
    pooled = tf.nn.avg_pool2d(tf.expand_dims(entropy_map, -1), ksize=window_size, strides=1, padding='SAME')
    flat = tf.reshape(pooled, [tf.shape(pooled)[0], -1])
    _, indices = tf.math.top_k(flat, k=top_k)
    return indices

def apply_local_attention(features, indices, d):
    batch_size = tf.shape(features)[0]
    patches = tf.image.extract_patches(images=features,
                                        sizes=[1, 7, 7, 1],
                                        strides=[1, 2, 2, 1],
                                        rates=[1, 1, 1, 1],
                                        padding='SAME')
    patches = tf.reshape(patches, [batch_size, -1, patches.shape[-1]])
    selected_patches = tf.gather(patches, indices, batch_dims=1)
    
    # Project and attend
    Wq = Dense(d)
    Wk = Dense(d)
    Wv = Dense(d)
    attention_outputs = []
    for patch in tf.unstack(selected_patches, axis=1):
        Q = Wq(patch)
        K = Wk(patch)
        V = Wv(patch)
        attn_scores = tf.nn.softmax(tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(d, tf.float32)))
        attn_output = tf.matmul(attn_scores, V)
        attention_outputs.append(attn_output)
    
    concat_attn = tf.concat(attention_outputs, axis=1)
    merged = Dense(d)(concat_attn)
    return merged

def build_elt(features, prob_map, d=64):
    entropy = compute_entropy(prob_map)
    indices = select_entropy_windows(entropy)
    Sattn = apply_local_attention(features, indices, d)

    # Residual + Feedforward
    W1 = Dense(4*d, activation='relu')(Sattn)
    W2 = Dense(d)(W1)
    FELT = Add()([W2, Sattn])
    return FELT
