# acr.py - STARE-Net module implementation

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LayerNormalization, Add, ReLU

def acr_refinement_layer(feature_map, filters=64):
    x = Conv2D(filters, kernel_size=3, padding='same')(feature_map)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = LayerNormalization()(x)
    return Add()([x, feature_map])  # Residual connection for refinement
