from os import listdir
from os.path import isfile, join
import pandas as pd

def teste(mypath):
      return [f for f in listdir(mypath) if isfile(join(mypath, f))]

def merge(x, y):

      x = x.reset_index(drop = True)
      y = y.reset_index(drop = True)

      #a = x.set_index("Date_time").merge(y, left_on=True)

      a = pd.concat([x, y], axis=1)

      #a.to_csv("saida.csv",sep=',',index=False)

      return pd.concat([x, y], axis=1)


file1 = "/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/dataset/Refine/weather_data_124.csv"
file2 = "/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/dataset/Refine/weather_data_124b.csv"
diry = "/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/dataset/teste"

listNames = teste("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/dataset/teste")

labels = ['Date_time', 'WY', 'Year', 'Month', 'Day', 'Hour', 'Minute']
print(listNames)
for index in range(len(listNames)):
      if index == 0:
            print(listNames[index])
            x = pd.read_csv(diry + "/" +listNames[index], sep=',')
            x.drop(columns=labels, inplace=True)
            print(x.head())
      else:
            print(listNames[index])
            y = pd.read_csv(file2, sep=",")
            y.drop(columns=labels, inplace=True)
            x = merge(x, y)
            print(x.head())


      
x.to_csv("weather_data_120_f.csv",sep=',',index=False)

