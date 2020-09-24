import numpy as np
import pandas as pd
from linear_regression_lib.linear_regression import *

#Takes data and splits it into training and testing data
x_values = np.array(pd.read_csv("resources/data.csv")[["km"]])
expected_values = np.array(pd.read_csv("resources/data.csv")[["price"]])
data = data_spliter(minmax_normalization(x_values), expected_values)
#More than 8000000 leads to overfitting, meaning too high cost on test data even if great cost on training data, less leads to underfitting
LR = linear_regression(np.zeros((x_values.shape[1] + 1, 1)), lambda_=0, n_cycle=8000000)
# LR.plot_(data[0], LR.predict_(data[0]), data[1], LR.cost_(LR.predict_(data[0]), data[1])) #Initial prediction line
print(LR.cost_(LR.predict_(data[0]), data[1]))
LR.fit_(data[0], data[1])
print(LR.cost_(LR.predict_(data[2]), data[3]))

LR.plot_(data[0], LR.predict_(data[0]), data[1], LR.cost_(LR.predict_(data[0]), data[1])) #prediction line for train data, spot underfitting
LR.plot_(data[2], LR.predict_(data[2]), data[3], LR.cost_(LR.predict_(data[2]), data[3])) #prediction line for test data, spot overfitting
pd.DataFrame(LR.theta, columns=["theta"]).to_csv("resources/theta.csv")
