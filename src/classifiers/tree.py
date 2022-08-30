from sklearn.tree import DecisionTreeClassifier



def arvore(train_dataset_x,test_dataset_x,train_dataset_y):

    arvRisco = DecisionTreeClassifier(criterion="entropy")

    model = arvRisco.fit(train_dataset_x,train_dataset_y)

    return model
    









