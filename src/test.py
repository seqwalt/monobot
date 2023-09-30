import numpy as np
from numpy import genfromtxt

try:
    P = genfromtxt('err_cov.txt', delimiter=',')
    print(P.shape)
except FileNotFoundError:
    print("Couldn't find covariance matrix file, using identity")
    X_sz = 21
    P = np.eye(X_sz)
