import sys
sys.path.append("..")

import numpy as np

from common import ModelBase


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

class LogisticRegression(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initWeights(self):
        self._weights = np.zeros((self._data_handler.numFeatures() + 1))

    def optimizeWeights(self, train_x, train_y, pred_y, learning_rate):
        error_y = pred_y - train_y
        self._weights -= learning_rate * ((error_y.dot(train_x)) / train_x.shape[0])

    def getPredictionMaths(self, data):
        return sigmoid((data * self._weights).sum(axis=1))
