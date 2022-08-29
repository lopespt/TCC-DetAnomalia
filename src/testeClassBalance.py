
from classbalance import imbalance

import pandas as pd

def balance():
    x = pd.read_csv("/home/kaike/code/tcc/src/x.csv")

    y = pd.read_csv("/home/kaike/code/tcc/src/y.csv")

    x = pd.DataFrame(x["values"])

    y = pd.DataFrame(y["values"])

    res_x, res_y = imbalance.randomBalance(x,y)

    print(res_x)

    print(res_y)

    return res_x, res_y