import wandb

import sklearn

import testPreProcessing as tp

import testeClassBalance as tc


from sklearn.model_selection import train_test_split

from classifiers import svm

wandb.init(project="my-test-project")

x,y = tc.balance()

a = tp.prepro(x)


X_train, X_test, y_train, y_test = train_test_split(a, y, test_size=0.3, random_state=0)

model = svm.vectorMachine(X_train, X_test, y_train)

wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test,  model_name="SVC")