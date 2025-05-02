# backbone.py - STARE-Net module implementation

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def build_backbone(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    layer_names = ['block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'out_relu']
    outputs = [base_model.get_layer(name).output for name in layer_names]
    backbone = Model(inputs=base_model.input, outputs=outputs, name="MobileNetV2_Backbone")
    return backbone
