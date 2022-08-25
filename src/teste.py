
from dados import read_csv
from classbalance import imbalance
from anomalyGenenrator import generator
import numpy as np
import pandas as pd


x, y = generator.anomaly("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/dataset/Weather%20Data/weather_data_124.txt", 1, 20)


print(x)
print(y)

pd.x.to_csv()
pd.y.to_csv()

