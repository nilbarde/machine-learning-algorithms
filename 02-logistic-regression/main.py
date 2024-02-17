import sys
sys.path.append("..")

import os
from common.data_handler import DataHandler
from model import LogisticRegression


DATASETS = "../datasets/"
dataset_name = "classification/coil_2000/"
dataset_path = os.path.join(DATASETS, dataset_name, "data.csv")

data = DataHandler(dataset_path, normalize=True)
LR = LogisticRegression(data_handler=data)

LR.train(num_epochs=50000, learning_rate=0.0004)
