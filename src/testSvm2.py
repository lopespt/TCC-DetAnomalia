import testPreProcessing as tp


import testeClassBalance as tc

import pandas as pd

from sklearn.model_selection import train_test_split

from classifiers import svm

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main():
    x,y = tc.balance()

    a = pd.DataFrame(tp.prepro(x))

    X_train, X_test, y_train, y_test = train_test_split(a, y, test_size=0.3, random_state=0)

    model,svc = svm.vectorMachine(X_train, X_test, y_train)

    print(confusion_matrix(y_test,model))
    print(classification_report(y_test,model))
    print(accuracy_score(y_test,model))

    return X_train, X_test, y_train, y_test, model,svc

