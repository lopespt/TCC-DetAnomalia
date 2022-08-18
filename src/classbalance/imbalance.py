from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np



def printHead(dataframe):
    print(dataframe.head())

def randomBalance(dataframe):
    rus = RandomUnderSampler(random_state=0)