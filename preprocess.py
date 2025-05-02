# preprocess.py - STARE-Net module implementation

import numpy as np
import cv2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def replace_outliers(data, threshold_low=1e-5, threshold_high=1e5):
    data = np.where((data < threshold_low) | (data > threshold_high), 0, data)
    return data

def handle_missing_values(data):
    imputer = SimpleImputer(strategy='mean')
    return imputer.fit_transform(data)

def min_max_normalize(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def gaussian_filter(image, ksize=(5, 5), sigma=1.0):
    return cv2.GaussianBlur(image, ksize, sigma)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = gaussian_filter(img)
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    return np.expand_dims(img, axis=-1)
