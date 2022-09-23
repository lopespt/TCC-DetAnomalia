from dados import read_csv
from classbalance import imbalance
from anomalyGenenrator import generator
import numpy as np
import pandas as pd
import pandas as ps

excel = pd.read_csv("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/dataset/Refine/weather_data_124.csv", nrows=200)

colunas = list(excel.columns)
colunas2 = colunas

for i in colunas:

    if i in ('Date_time', 'WY', 'Year', 'Month', 'Day', 'Hour', 'Minute'):
        colunas2.remove(i)


colunas2.remove('WY')
colunas2.remove('Month')
colunas2.remove('Hour')
print(colunas2)


#x, y = generator.anomaly("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/dataset/Weather%20Data/weather_data_124.txt", 2, 20, "T_a")

#print(type(x))    
#print(type(y)) 

#x = pd.DataFrame(x)
#y = pd.DataFrame(y)

#print(x.head())
#print(y.head())

#x.to_csv("x.csv")
#y.to_csv("y.csv")



for i in colunas2:
    x, y = generator.anomaly("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/dataset/Refine/weather_data_124.csv", 2, 20, i)

    x  = pd.DataFrame(x)
    

    x.to_csv(i + "_x.csv")

    #y = pd.DateFrame(y)
    y.to_csv(i + "_y.csv")