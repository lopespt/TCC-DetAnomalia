from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



def vectorMachine(train_dataset_x,train_dataset_y):
    svm = SVC(kernel="linear", random_state=1, C=1.0, probability=True)
    model = svm.fit(train_dataset_x,train_dataset_y)

    return model,svm
    
