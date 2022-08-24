
from dados import read_csv
from classbalance import imbalance
from anomalyGenenrator import generator
import numpy as np


print(generator.anomaly("/home/kaike/code/tcc/dataset/Weather%20Data/weather_data_124.txt", 1, 20))

