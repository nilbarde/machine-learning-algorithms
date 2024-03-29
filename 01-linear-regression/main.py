import sys
sys.path.append("..")

import os
from common.data_handler import DataHandler
from model import LinearRegression


DATASETS = "../datasets/"
dataset_name = "regression/energy_efficiency/y1/"
dataset_path = os.path.join(DATASETS, dataset_name, "data.csv")

data = DataHandler(dataset_path, normalize=True)
LR = LinearRegression(data_handler=data)

LR.train(num_epochs=2000000)
