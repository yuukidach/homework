# -*- coding utf-8 -*-

import pandas as pd
import numpy as np
import linear_classification as lc

def get_iris_data():
    dataset = pd.read_csv("iris.csv", header=0)

    x1 = dataset.iloc[0:50, [0, 2]].values
    x2 = dataset.iloc[50:100, [0, 2]].values
    y1 = dataset.iloc[0:50, 4].values
    y2 = dataset.iloc[50:100, 4].values

    x = np.vstack((x1, x2))
    y = np.append(y1, y2)
    y = np.where(y=='setosa', 1, -1)

    return x, y

if __name__ == "__main__":
    '''
    x = np.array([
    [-2, 4],
    [4, 1],
    [1, 6],
    [2, 4],
    [6, 2],])

    y = np.array([-1,-1,1,1,1]) 
    '''

    x, y = get_iris_data()
    print(x, y)
    print(lc.perception_plt(x, y, 0.4, 10))


