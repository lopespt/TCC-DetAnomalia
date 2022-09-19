from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



def arvore(train_dataset_x,train_dataset_y):

    arvRisco = DecisionTreeClassifier(criterion="entropy")

    model = arvRisco.fit(train_dataset_x,train_dataset_y)

    return model, arvRisco


def aleArvore(train_dataset_x,train_dataset_y):


    randomForest = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
    model = randomForest.fit(train_dataset_x, train_dataset_y)

    return model, randomForest









