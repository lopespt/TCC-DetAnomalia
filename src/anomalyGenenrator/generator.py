
from random import random
import re
import pandas as pd
import numpy as np
from math import ceil
import random
from tqdm import tqdm

#Preciso gerar anomalias na base e indentificalas
#Elas precisam estar numa escala definida no tcc seguindo 2 parametros (intensidade, repetição da base)
#essas 2 variaveis vão de 20 a 80 %
#é necessario ter 4 variaveis a base de treino x e y onde x é a base bruta e y é o valor se é anomalia ou não
# e a base de teste x e y para comparar e fazer as metricas




def anomaly(csv, intensidade, repeticao):
    ids = []

    dataframe = pd.read_csv(csv)
    #a função ceil arredonda para cima
    nElementos = ceil(dataframe.size * repeticao / 100)
    for i in range(dataframe.size):
        ids.append(i)

    y = random.sample(ids,nElementos)

    lista = []
    alt = []


    for index in tqdm(dataframe.index):
        #print(dataframe["T_a"][index])
        if index in y:
            alt.append(1)
            ano = desPadrao(dataframe,intensidade) + i
            lista.append(ano)
        else:
            alt.append(0)
            lista.append(index)

    return alt



def desPadrao (dataframe, intensidade):
    
    x = dataframe["T_a"].values.tolist()

    xdes = np.std(x)
    #print(xdes)

    return intensidade * xdes