from sklearn.model_selection import GridSearchCV

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

params = {
      "kernel" : ["linear", "rbf"]
}

model = SVC()


def seletor():
      return GridSearchCV(model, params, cv=10, scoring="accuracy")