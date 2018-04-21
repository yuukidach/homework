# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt 

tmp = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],

])

y = np.array([-1,-1,1,1,1])

def perception(x, y):
    w = np.zeros(x.shape[1])
    eta = 1
    epochs = 20

    for t in range(epochs):
        for i, xi in enumerate(x):
            if (np.dot(xi, w)*y[i]) <= 0:
                w = w + eta*xi*y[i]

    return w


def perception_plt(x,y):
    w = np.zeros(x.shape[1])
    eta = 1
    epochs = 20
    errs = []

    for t in range(epochs):
        total_err = 0
        for i, xi in enumerate(x):
            decision = np.dot(xi, w)*y[i]
            if decision <= 0:
                w = w + eta*xi*y[i]
                total_err += decision
        errs.append(total_err*-1)

    plt.plot(errs)
    plt.xlabel('epoch')
    plt.ylabel('total loss')
    
    return w

if __name__ == "__main__":
    w = perception_plt(tmp, y)
    print(w)
    plt.show()
