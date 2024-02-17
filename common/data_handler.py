import os
import pandas as pd
import numpy as np


class DataHandler():
    def __init__(self, path, col_target="target", split_ratio=0.8, shuffle=False, normalize=False, normalize_method="min_max", features_to_drop=[]):
        self.data = pd.read_csv(path)
        self.col_target = col_target
        self.split_ratio = split_ratio
        self.shuffle = shuffle
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.features_to_drop = features_to_drop

        self.prepare()

    def prepare(self):
        self.features = [col for col in self.data.columns.to_list() if col not in (self.features_to_drop + [self.col_target])]
        self.data = self.data[self.features + [self.col_target]]

        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

        self.num_samples = self.data.shape[0]
        self.num_samples_train = int(self.num_samples * self.split_ratio)
        self.num_samples_test = self.num_samples - self.num_samples_train

        train = self.data.iloc[:self.num_samples_train]
        test = self.data.iloc[self.num_samples_train:]

        self.makePreprocess(train)

        self.train_x, self.train_y = self.preprocess(train, target=True)
        self.test_x, self.test_y = self.preprocess(test, target=True)

    def makePreprocess(self, data):
        if self.normalize:
            if self.normalize_method == "min_max":
                self.normalize_params = {}
                self.normalize_params["min"] = data[self.features].to_numpy().min(axis=0)
                self.normalize_params["max"] = data[self.features].to_numpy().max(axis=0)
            else:
                raise ValueError(f"Passed normalization method {self.normalize_method} is not currently supported")

    def preprocess(self, data, target=False):
        data_x = data[self.features].to_numpy()
        if self.normalize:
            if self.normalize_method == "min_max":
                data_x = (data_x - self.normalize_params["min"]) / (self.normalize_params["max"] -self.normalize_params["min"])
        bias = np.ones((data_x.shape[0], 1))
        data_x = np.hstack((bias, data_x))
        if not target:
            return data_x
        data_y = data[self.col_target].to_numpy()
        return data_x, data_y

    def getTrain(self):
        return self.train_x, self.train_y

    def getTest(self):
        return self.test_x, self.test_y

    def numFeatures(self):
        return len(self.features)
