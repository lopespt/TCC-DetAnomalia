import pandas as pd

import numpy as np


from geneticAlgoritm import grid

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import testSvm2 as ts

X_train, X_test, y_train, y_test, model, svc = ts.main()
print(type(svc))
a = grid.seletor()

selectors = a.fit(X_train, y_train)
print(selectors)
y_predicy_ga = a.predict(X_test)
print(y_predicy_ga)
print(accuracy_score(y_test,y_predicy_ga))
