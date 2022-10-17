import pandas as pd


array = [ 0.84305051,  0.10634407, -0.71159748, -0.66952699]


dataset=pd.read_csv("/home/kaike/code/tcc/src/dados/experimento3/randforest/3_20/124_3_20_x_-3.csv", sep=",")


print(dataset.head())

dataset.drop(columns="-1.8", inplace=True)
#print(dataset.head())
dataset.to_csv("124_3_20_x_-4.csv",sep=',',index=False)