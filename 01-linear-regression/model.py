import sys
sys.path.append("..")

import numpy as np

from common import ModelBase


class LinearRegression(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initWeights(self):
        self.weights = np.zeros((self.data_handler.numFeatures() + 1))

    def optimizeWeights(self, train_x, train_y, pred_y, learning_rate):
        error_y = pred_y - train_y
        self.weights -= learning_rate * ((error_y.dot(train_x)) / train_x.shape[0])

    def getPredictionMaths(self, data):
        return (data * self.weights).sum(axis=1)
