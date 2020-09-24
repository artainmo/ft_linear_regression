import numpy as np
import pandas as pd
import sys
from linear_regression_lib.linear_regression import *

# Program takes one argument, namely the mileage, and the program prints its associated price prediction
def mileage_normalization():
    x_values = np.append(np.array(pd.read_csv("resources/data.csv")[["km"]]), float(sys.argv[1]))
    return minmax_normalization(x_values)[x_values.shape[0] - 1]


if __name__=="__main__":
    thetas = np.array(pd.read_csv("resources/theta.csv")[["theta"]])
    prediction = thetas[0] + (thetas[1] * mileage_normalization())
    print(prediction)
