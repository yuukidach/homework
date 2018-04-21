# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt 


def perception(x, y, eta, epochs):
    w = np.ones(x.shape[1]+1)      # 加上常数项

    for t in range(epochs):
        for i, xi in enumerate(x):
            xi = np.append(xi, 1)
            if (np.dot(xi, w)*y[i]) <= 0:
                w = w + eta*xi*y[i]

    return w


def perception_plt(x, y, eta, epochs):
    w = np.ones(x.shape[1]+1)
    errs = []

    for t in range(epochs):
        total_err = 0
        for i, xi in enumerate(x):
            xi = np.append(xi, 1)
            decision = np.dot(xi, w)*y[i]
            if decision <= 0:
                w = w + eta*xi*y[i]
                total_err += decision
        errs.append(total_err*-1)

    plt.plot(errs)
    plt.xlabel('epoch')
    plt.ylabel('total loss')
    plt.show()
    
    return w


