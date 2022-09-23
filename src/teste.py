import pandas as pd

import numpy as np


x = np.genfromtxt("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/src/x_test.csv", delimiter=",")
y = np.genfromtxt("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/src/y_test.csv", delimiter=",")
#print(x)
#print(y)

a = np.array([10, 100, 1000])

lines, columns = x.shape

for i in range(columns):
      for j in range(lines):
            print(x[j][i] * a[i])