from dados import read_csv
from classbalance import imbalance
from anomalyGenenrator import generator
import numpy as np
import pandas as pd
import pandas as ps

excel = pd.read_csv(
    "/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/dataset/Refine/weather_data_124.csv")

colunas = list(excel.columns)
colunas2 = colunas
labels = ['Date_time', 'WY', 'Year', 'Month', 'Day', 'Hour', 'Minute']


excel.drop(columns=labels, inplace=True)

#x, y = generator.anomaly("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/dataset/Weather%20Data/weather_data_124.txt", 2, 20, "T_a")

# print(type(x))
# print(type(y))

#x = pd.DataFrame(x)
#y = pd.DataFrame(y)

# print(x.head())
# print(y.head())

# x.to_csv("x.csv")
# y.to_csv("y.csv")

x, y = generator.anomaly(excel, 2, 20)

x = pd.DataFrame(x)

x.to_csv("124_x.csv", header=False, index=False)

#y = pd.DateFrame(y)
y.to_csv("124_y.csv", header=False, index=False)
