import numpy as np
import math

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

def relu(x):
    return max(0.0, x)