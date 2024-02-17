import numpy as np

from .data_handler import DataHandler


class ModelBase():
    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler
        self.initWeights()

    def train(self, learning_rate: float=0.00001, num_epochs: int=1000):
        train_x, train_y = self.data_handler.getTrain()
        test_x, test_y = self.data_handler.getTest()

        for epoch in range(num_epochs):
            train_pred = self.getPredictionMaths(train_x)
            train_error = self.getError(train_pred, train_y)
            self.optimizeWeights(train_x=train_x, train_y=train_y, pred_y=train_pred, learning_rate=learning_rate)

            test_pred = self.getPredictionMaths(test_x)
            test_error = self.getError(test_pred, test_y)

            if ((epoch+1) % (num_epochs/100) == 0):
                print(f"Epoch: {epoch+1}/{num_epochs}; train error: {train_error}; test error: {test_error}", end="\r")
            if ((epoch+1) % (num_epochs/10) == 0):
                print(f"Epoch: {epoch+1}/{num_epochs}; train error: {train_error}; test error: {test_error}")

    def getPrediction(self, data):
        data = self.data_handler.preprocess(data)
        return self.getPredictionMaths(data)

    def getPredictionMaths(self, data):
        return (data * self.weights).sum(axis=1)

    def getError(self, pred, gt):
        error_y = (pred - gt)
        error_sum = ((error_y) ** 2) / pred.shape[0]
        return np.round(error_sum.sum(), 4)
