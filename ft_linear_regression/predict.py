import numpy as np
import pandas as pd
import sys
from linear_regression_lib import *

def mileage_normalization():
    mileage = input("Mileage to predict price on: ")
    x_values = np.append(np.array(pd.read_csv("resources/data.csv")[["km"]]), float(mileage)) #Adding mileage to other x_values to normalize
    return minmax_normalization(x_values)[x_values.shape[0] - 1] #Only return the last value or normalized mileage


if __name__=="__main__":
    if len(sys.argv) != 1:
        print("Error Argument")
        exit();
    thetas = np.array(pd.read_csv("resources/theta.csv")[["theta"]])
    prediction = thetas[0] + (thetas[1] * mileage_normalization())
    print("Predicted price for given mileage: " + str(prediction[0]))
