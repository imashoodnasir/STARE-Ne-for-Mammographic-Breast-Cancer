# ensemble.py - STARE-Net module implementation

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class MetaEnsembleClassifier:
    def __init__(self, weights=(0.5, 0.5)):
        self.wsvm = SVC(kernel='rbf', probability=True, class_weight='balanced')
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.weights = weights  # (w_svm, w_knn)

    def fit(self, X_train, y_train):
        self.wsvm.fit(X_train, y_train)
        self.knn.fit(X_train, y_train)

    def predict(self, X):
        p_svm = self.wsvm.predict_proba(X)
        p_knn = self.knn.predict_proba(X)
        ensemble_prob = self.weights[0] * p_svm + self.weights[1] * p_knn
        return np.argmax(ensemble_prob, axis=1)

    def predict_proba(self, X):
        p_svm = self.wsvm.predict_proba(X)
        p_knn = self.knn.predict_proba(X)
        return self.weights[0] * p_svm + self.weights[1] * p_knn
