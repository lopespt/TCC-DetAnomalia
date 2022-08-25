from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np

#https://www.youtube.com/watch?v=4SivdTLIwHc&ab_channel=DataProfessor

def printHead(dataframe):
    print(dataframe.head())


#Undersampling
def randomBalance(x,y):
    rus = RandomUnderSampler(random_state=0)
    x_res, y_res = rus.fit_resample(x,y)
    return x_res, y_res