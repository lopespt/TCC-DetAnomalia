from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

arvRisco = DecisionTreeClassifier(criterion="entropy")

def arvore(train_dataset_x,train_dataset_y):


    model = arvRisco.fit(train_dataset_x,train_dataset_y)

    return model, arvRisco


def aleArvore(train_dataset_x,train_dataset_y):


    randomForest = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
    model = randomForest.fit(train_dataset_x, train_dataset_y)

    return model, randomForest









