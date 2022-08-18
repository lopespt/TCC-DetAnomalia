from http.client import ImproperConnectionState
from sklearn.linear_model import LogisticRegression

def LogReg(train_dataset_x,test_dataset_x,train_dataset_y):

    logistic = LogisticRegression(random_state=1)
    alg = logistic.fit(train_dataset_x,train_dataset_y)

    return alg