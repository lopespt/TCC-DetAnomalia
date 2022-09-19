import numpy as np
import pandas as pd

from fusionData import *
from fusionData.fusion import conversion, creatingMatrix

import testPreProcessing as tp


import testeClassBalance as tc 


x, y = tc.balance()


a = conversion(x)

b = creatingMatrix(a)

print(b)
