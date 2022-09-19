import numpy as np

import pandas as pd


nClassificadores = 4

def conversion(dataframe):

      return dataframe.to_numpy()


def creatingMatrix(npa):
      tamanho = npa.shape
      x = tamanho[0]
      y = tamanho[1]
      pesos =np.ones((x-2, y))

      return pesos