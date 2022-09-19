import pandas as pd
import numpy as np

import testSvm2 as ts

from classifiers import tree, logisticReg, svm

import testPreProcessing as tp

import wandb

from metrics import allMetrics
import testeClassBalance as tc

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



wandb.init(project="my-test-project")
x,y = tc.balance()

a = pd.DataFrame(tp.prepro(x))

X_train, X_test, y_train, y_test = train_test_split(a, y.values.ravel(), test_size=0.3, random_state=0)
#y_train = y_train.values.ravel()
#y_test = y_test.values.ravel()
wandb.log({"X": a,"y": y,"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test":y_test})
svm_model,svc = svm.vectorMachine(X_train, y_train)
labels = np.array(["anomalia", "n_anomalia"])

print("--------------SVM----------------------")
svm_predict = svm_model.predict(X_test)
print(confusion_matrix(y_test,svm_predict))
print(classification_report(y_test,svm_predict))
print(accuracy_score(y_test,svm_predict))

wandb.sklearn.plot_calibration_curve(svm_model, a, y, "SupportVectorMachine")
wandb.sklearn.plot_confusion_matrix(y_test, svm_predict, labels)
wandb.sklearn.plot_outlier_candidates(svm_model, a, y)
#wandb.sklearn.plot_silhouette(svm_model, X_train, labels)
wandb.sklearn.plot_learning_curve(svm_model, a, y)
wandb.sklearn.plot_roc(y_test, svm_model.predict_proba(X_test), labels)
#wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
wandb.sklearn.plot_summary_metrics(svm_model, X_train, y_train, X_test, y_test)
#wandb.sklearn.plot_feature_importances(svm_model, []) 
print("-------------------------------")


log_model,log = logisticReg.LogReg(X_train, y_train)


print("--------------Logictic----------------------")
print(log_model.coef_)
log_predict = log_model.predict(X_test)
print(log_predict)
print(confusion_matrix(y_test,log_predict))
print(classification_report(y_test,log_predict))
print(accuracy_score(y_test,log_predict))


wandb.sklearn.plot_calibration_curve(log_model, a, y, "LogisticRegression")
wandb.sklearn.plot_confusion_matrix(y_test, log_predict, labels)
wandb.sklearn.plot_outlier_candidates(log_model, a, y)
#wandb.sklearn.plot_silhouette(svm_model, X_train, labels)
wandb.sklearn.plot_learning_curve(log_model, a, y)
wandb.sklearn.plot_roc(y_test, log_model.predict_proba(X_test), labels)
#wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
wandb.sklearn.plot_summary_metrics(log_model, X_train, y_train, X_test, y_test)

print("-------------------------------")

dec_model, dec = tree.arvore(X_train, y_train)

print("--------------ArvDecisão----------------------")
#print(dec_model.coef_)
dec_predict = dec_model.predict(X_test)
print(dec_predict)
print(confusion_matrix(y_test,dec_predict))
print(classification_report(y_test,dec_predict))
print(accuracy_score(y_test,dec_predict))
print("-------------------------------")

ram_model, ram = tree.aleArvore(X_train, y_train)

print("--------------RandomArvDecisão----------------------")
#print(dec_model.coef_)
ram_predict = ram_model.predict(X_test)
print(ram_predict)
print(confusion_matrix(y_test,ram_predict))
print(classification_report(y_test,ram_predict))
print(accuracy_score(y_test,ram_predict))
print("-------------------------------")




#wandb.sklearn.plot_classifier(svm_model,X_train, X_test, y_train, y_test,svm_predict,svm_model.predict_proba(X_test), labels ,is_binary=True, model_name="SupportVectorMachine", feature_names=["anomalia", "n_anomalia"])


wandb.sklearn.plot_calibration_curve(log_model, a, y, "LogisticRegression")
wandb.sklearn.plot_confusion_matrix(y_test, log_predict, labels)
wandb.sklearn.plot_outlier_candidates(log_model, a, y)
#wandb.sklearn.plot_silhouette(svm_model, X_train, labels)
wandb.sklearn.plot_learning_curve(log_model, a, y)
wandb.sklearn.plot_roc(y_test, log_model.predict_proba(X_test), labels)
#wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
wandb.sklearn.plot_summary_metrics(log_model, X_train, y_train, X_test, y_test)
#wandb.sklearn.plot_feature_importances(svm_model, []) 