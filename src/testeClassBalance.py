from classbalance import imbalance


from os import listdir
from os.path import isfile, join
#import pandas as pd
import numpy as np

from tqdm import tqdm
from preprocessing import proc
def teste(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

def merge(dirr,dirry):
    listName = teste(dirr)
    listName2 = teste(dirry)
    #a de auxiliar
    a = []
    for i in tqdm(listName):
        if len(a) == 0:
            x = np.genfromtxt(dirr + "/" + i, delimiter=",")
            a = x[:, 1][np.logical_not(np.isnan(x[:, 1]))]
            a = a.reshape(-1,1)
            pass
        else:
            x = np.genfromtxt(dirr + "/" + i, delimiter=",")
            x = x[:, 1][np.logical_not(np.isnan(x[:, 1]))]
            x = x.reshape(-1,1)
            a = np.append(a, x, axis=1)

    x = a.astype(np.float32)
    #x = a
    a = []
    for i in tqdm(listName2):
        if len(a) == 0:
            y = np.genfromtxt(dirry + "/" + i, delimiter=",")
            a = y[:, 1][np.logical_not(np.isnan(y[:, 1]))]
            #a = a.reshape(-1,1)
        else:
            y = np.genfromtxt(dirry + "/" + i, delimiter=",")
            y = y[:, 1][np.logical_not(np.isnan(y[:, 1]))]
            for j in range(len(a)):
                if a[j] == 0 and y[j] == 0:
                    a[j] = 0
                else:
                    a[j] = 1
        
    y = a.astype(np.int8) 
    return x,y.reshape(-1,1)



def balance():
    x, y = merge("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/src/xfiles","/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/src/yfiles")
    res_x, res_y = imbalance.randomBalance(x,y)
    res_x = proc.scalar(res_x)
    print(res_x)
    #np.savetxt("x_out.csv",res_x,delimiter=",",fmt='%f')
    print(res_y)
    #np.savetxt("y_out.csv",res_y,delimiter=",",fmt='%f')
    return res_x, res_y

#x, y = balance()
#a, b = merge("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/src/xfiles","/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/src/yfiles")
