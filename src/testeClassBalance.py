from classbalance import imbalance
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
#import pandas as pd
import numpy as np
import pandas as pd

from tqdm import tqdm
#from preprocessing import proc
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
    #x, y = merge("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/src/xfiles","/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/src/yfiles")
    x = np.genfromtxt("/home/tccanomalia/tcc/TCC-DetAnomalia/src/dados/experimento1/2Intensidade_20Porcentagem_x.csv", delimiter=",")
    #a = np.genfromtxt("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/src/dados/experimento2/weather_data_jd_f.csv", delimiter=",")
    y = np.genfromtxt("/home/tccanomalia/tcc/TCC-DetAnomalia/src/dados/experimento1/2Intensidade_20Porcentagem_y.csv", delimiter=",")
    #np.savetxt("y_gui.csv",y,delimiter=",",fmt='%f')
    #print(np.isnan(a))
    res_x, res_y = imbalance.randomBalance(x,y)
    #print(np.isnan(res_x))
    X_train, X_test_temp, y_train, y_test_temp = train_test_split(res_x, res_y, test_size=0.3, random_state=1)
    print("Xtrain: " +  str(np.shape(X_train)[0]))
    print("ytrain: " +  str(np.shape(y_train)[0]))
    X_test_classificator, X_test_ga,y_test_classificator, y_test_ga = train_test_split(X_test_temp, y_test_temp, test_size=1/3, random_state=1)
    print("X_test_ga: " +  str(np.shape(X_test_ga)[0]))
    print("y_test_ga: " +  str(np.shape(y_test_ga)[0]))
    print("x_test: " +  str(np.shape(X_test_classificator)[0]))
    print("y-test: " + str(np.shape(y_test_classificator)[0]))
    s = preprocessing.MinMaxScaler().fit(X_train)
    X_train = s.transform(X_train)
    X_test_classificator = s.transform(X_test_classificator)
    X_test_ga = s.transform(X_test_ga)
    #np.savetxt("x_out.csv",res_x,delimiter=",",fmt='%f')
    #np.savetxt("y_out.csv",res_y,delimiter=",",fmt='%f')
    return X_train, X_test_classificator, y_train, y_test_classificator,X_test_ga,y_test_ga

#x, y = balance()
#a, b = merge("/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/src/xfiles","/home/kaike/Documents/Code/Tcc/TCC-DetAnomalia/src/yfiles")
X_train, X_test_classificator, y_train, y_test_classificator,X_test_ga,y_test_ga = balance()

#print(X_train)