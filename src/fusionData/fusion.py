import numpy as np

import pandas as pd


nClassificadores = 4

def conversion(dataframe):

      return dataframe.to_numpy()


def creatingMatrix(npa):
      linhas, colunas = npa.shape
      pesos = np.ones(colunas)

      return pesos

def potara(valor, pesos):
      return valor * pesos