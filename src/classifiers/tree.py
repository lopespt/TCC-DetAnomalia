from sklearn.tree import DecisionTreeClassifier, RandomForestClassifier



def arvore(train_dataset_x,test_dataset_x,train_dataset_y):

    arvRisco = DecisionTreeClassifier(criterion="entropy")

    model = arvRisco.fit(train_dataset_x,train_dataset_y)

    return model


def aleArvore(train_dataset_x,train_dataset_y,test_dataset_x):


    randomForest = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
    randomForest.fit(train_dataset_x, train_dataset_y)

    return randomForest.predict(test_dataset_x)









