import os
import pandas as pd
import numpy as np


class DataHandler():
    def __init__(self, path, col_target="target", split_ratio=0.8, shuffle=False, normalize=False, normalize_method="min_max", features_to_drop=[]):
        self._data = pd.read_csv(path)
        self._col_target = col_target
        self._split_ratio = split_ratio
        self._shuffle = shuffle
        self._normalize = normalize
        self._normalize_method = normalize_method
        self._normalize_params = {}
        self._features_to_drop = features_to_drop

        self.prepare()

    def prepare(self):
        self._feature_cols = [col for col in self._data.columns.to_list() if col not in (self._features_to_drop + [self._col_target])]
        self._data = self._data[self._feature_cols + [self._col_target]]

        if self._shuffle:
            self._data = self._data.sample(frac=1).reset_index(drop=True)

        self._num_samples = self._data.shape[0]
        num_samples_train = int(self._num_samples * self._split_ratio)

        train = self._data.iloc[:num_samples_train]
        test = self._data.iloc[num_samples_train:]

        self.makePreprocess(train)

        self._train_x, self._train_y = self.preprocess(train, target=True)
        self._test_x, self._test_y = self.preprocess(test, target=True)

    def makePreprocess(self, data):
        if self._normalize:
            if self._normalize_method == "min_max":
                self._normalize_params["min"] = data[self._feature_cols].to_numpy().min(axis=0)
                self._normalize_params["max"] = data[self._feature_cols].to_numpy().max(axis=0)
            else:
                raise ValueError(f"Passed normalization method {self._normalize_method} is not currently supported")

    def preprocess(self, data, target=False):
        data_x = data[self._feature_cols].to_numpy()
        if self._normalize:
            if self._normalize_method == "min_max":
                data_x = (data_x - self._normalize_params["min"]) / (self._normalize_params["max"] -self._normalize_params["min"])
        bias = np.ones((data_x.shape[0], 1))
        data_x = np.hstack((bias, data_x))
        if not target:
            return data_x
        data_y = data[self._col_target].to_numpy()
        return data_x, data_y

    def getTrain(self):
        return self._train_x, self._train_y

    def getTest(self):
        return self._test_x, self._test_y

    def numFeatures(self):
        return len(self._feature_cols)
