from math import ceil
import pandas as pd

import numpy as np
np.set_printoptions(precision=2,suppress=True)

datasetPandas = pd.read_csv("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/dataset/Refine/weather_data_124.csv")
#datasetNumpy = np.genfromtxt("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/dataset/Refine/weather_data_124.csv", delimiter=",", skip_header=1, usecols= (7, 8, 9, 10, 11, 12, 13), defaultfmt='%f')
labels = ['Date_time', 'WY', 'Year', 'Month', 'Day', 'Hour', 'Minute']

datasetPandas.drop(columns=labels,inplace=True)

#y = np.genfromtxt("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/src/y_test.csv", delimiter=",")
#print(x)
#print(y)

a = np.array([10, 100, 1000])



datasetNumpy = datasetPandas.to_numpy()
print(datasetNumpy)
print(datasetPandas)


lines, columns = datasetNumpy.shape

print(lines)
print(columns)
np.random.seed(1)
teste1 = np.random.random(size=columns)
np.random.seed(1)
teste2 = np.random.random(size=columns)

print(datasetPandas.iloc[:,6])

print(teste1)
print(teste2)

# for i in range(columns):
#       print("\n")
#       for j in range(10):
#             print(datasetNumpy[j][i])



np.random.seed(1)
    #ids = []
    
    #a função ceil arredonda para cima

nElementos = ceil(datasetPandas.size * 50 / 100)
ids = np.random.randint(lines, size=nElementos)
print(ids)

np.random.seed(2)

def desPadrao (dataframe, coluna):
    
    x = dataframe[coluna].values.tolist()

    xdes = np.std(x)
    #print(xdes)

    return xdes


for i in range(linhas):
      if i in (ids):
            

