from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def score(predics, test_dataset_y):
    return accuracy_score(predics,test_dataset_y)

def classification(predicts, test_dataset_y):
    return classification_report(test_dataset_y, predicts)
