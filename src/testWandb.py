import wandb

import sklearn

import testPreProcessing as tp

import testeClassBalance as tc

import testSvm2 as ts

from sklearn.model_selection import train_test_split

from classifiers import svm


y_test, predics = ts.main()


class_names = ["VT", "VN"]


cm = wandb.plot.confusion_matrix(
    y_true=y_test,
    preds=predics,
    class_names=class_names)
    
wandb.log({"conf_mat": cm})