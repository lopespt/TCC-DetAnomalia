import pandas as pd
import numpy as np

import testeClassBalance as tc

from preprocessing import proc

x,y = tc.balance()

def prepro(dataset):

    return proc.scalar(dataset)


a = pd.DataFrame(prepro(x))

print(a.head())
