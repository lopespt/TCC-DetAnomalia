from dados import read_csv
from classbalance import imbalance
from anomalyGenenrator import generator
import numpy as np
import pandas as pd


x, y = generator.anomaly("/home/kaike/code/tcc/dataset/Weather%20Data/weather_data_124.txt", 2, 20)

print(type(x))    
print(type(y)) 

x = pd.DataFrame(x)
y = pd.DataFrame(y)

print(x.head())
print(y.head())

x.to_csv("x.csv")
y.to_csv("y.csv")

