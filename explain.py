# explain.py - STARE-Net module implementation

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import shap

def compute_gradcam(model, image, layer_name="out_relu"):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(tf.expand_dims(image, axis=0))
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output[0]), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / tf.reduce_max(heatmap)
    return heatmap.numpy()

def compute_shap(model, background, test_sample):
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(test_sample)
    shap.image_plot(shap_values, test_sample)

def compute_osa(model, image, patch_size=32):
    h, w, c = image.shape
    heatmap = np.zeros((h, w))
    base_pred = model.predict(tf.expand_dims(image, axis=0))[0]
    base_class = np.argmax(base_pred)

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            img_copy = image.copy()
            img_copy[i:i+patch_size, j:j+patch_size, :] = 0
            pred = model.predict(tf.expand_dims(img_copy, axis=0))[0]
            prob_drop = base_pred[base_class] - pred[base_class]
            heatmap[i:i+patch_size, j:j+patch_size] = prob_drop

    heatmap = heatmap / np.max(heatmap)
    return heatmap
