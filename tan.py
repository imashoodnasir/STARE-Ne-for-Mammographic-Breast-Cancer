# tan.py - STARE-Net module implementation

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Add, ReLU

def fuse_hmt_elt_outputs(global_feat, local_feat, d=64):
    # Flatten and project both inputs to same dimension
    global_flat = tf.reshape(global_feat, [tf.shape(global_feat)[0], -1, global_feat.shape[-1]])
    local_flat = tf.reshape(local_feat, [tf.shape(local_feat)[0], -1, local_feat.shape[-1]])

    global_proj = Dense(d)(global_flat)
    local_proj = Dense(d)(local_flat)

    fused = Add()([global_proj, local_proj])
    fused = Dense(d)(ReLU()(fused))

    return fused  # This becomes the final representation for ACR
