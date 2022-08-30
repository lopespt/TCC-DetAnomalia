import testPreProcessing as tp

import testeClassBalance as tc

import pandas as pd

from sklearn.model_selection import train_test_split

from classifiers import tree as tr

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



from sklearn import tree

from matplotlib import pyplot as plt


x,y = tc.balance()

a = pd.DataFrame(tp.prepro(x))

X_train, X_test, y_train, y_test = train_test_split(a, y, test_size=0.3, random_state=0)

model = tr.arvore(X_train, X_test, y_train)

model.predict(X_test)

fig = plt.figure(figsize=(25,20))

_ = plt.figure(tree.plot_tree(model))
