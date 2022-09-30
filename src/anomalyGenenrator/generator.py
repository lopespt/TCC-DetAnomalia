
from random import random
import re
from unittest import skip
import pandas as pd
import numpy as np
from math import ceil
import random
from tqdm import tqdm

import threading

#Preciso gerar anomalias na base e indentificalas
#Elas precisam estar numa escala definida no tcc seguindo 2 parametros (intensidade, repetição da base)
#essas 2 variaveis vão de 20 a 80 %
#é necessario ter 4 variaveis a base de treino x e y onde x é a base bruta e y é o valor se é anomalia ou não
# e a base de teste x e y para comparar e fazer as metricas
def desPadrao (dataframe, coluna):
    
    x = dataframe[coluna].values

    xdes = np.std(x)
    #print(xdes)

    return xdes

def apply(dataframe, intensidade, ids):
    np.random.seed(2)
    lista = []
    alt = []
    df = dataframe
    
    lines, colums = df.shape
   
    #y = random.sample(ids,nElementos)

    for index in tqdm(dataframe.index):
        #print(dataframe["T_a"][index])
        
        if index in ids:
            alt.append(1)
            coluna = np.random.choice(df.columns.values, size=1)
            multiplo = intensidade * desPadrao(df, coluna)
            df.loc[index,coluna] = round(multiplo + df[coluna].iloc[index], 1)
            #lista.append(ano)
        else:
            alt.append(0)
            #lista.append(dataframe[coluna].iloc[index])
    return df, alt

def anomaly(dataframe, intensidade, repeticao):
    
    np.random.seed(1)
    #ids = []

    #dataframe = pd.read_csv(csv) #on_bad_lines="skip)
    
    #a função ceil arredonda para cima
    lines, colums = dataframe.shape

    nElementos = ceil(dataframe.size * repeticao / 100)
    ids = np.random.randint(lines, size=nElementos)
    # for i in tqdm(range(dataframe.size)):
    #     ids.append(i)

    


    lista, y = apply(dataframe, intensidade, ids)

    alt = pd.DataFrame(y)

    return lista, alt


