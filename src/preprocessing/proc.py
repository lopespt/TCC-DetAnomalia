
from sklearn import preprocessing

import numpy as np

import pandas as pd


def scalar(dataset):

    return preprocessing.MinMaxScaler().fit_transform(dataset)