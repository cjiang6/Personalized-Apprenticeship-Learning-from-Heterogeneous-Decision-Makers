"""
create a dt that should achieve .99% accuracy on training
"""

import sys
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from utils.naive_utils import create_new_dataset
from sklearn.metrics import accuracy_score
from sklearn import tree
import copy
sys.path.insert(0, '../')

# Naive

def train():
    num_schedules = 150  
    total_loss_array = []

    load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
        num_schedules) + 'dist_early_hili_naive.pkl'

    data = pickle.load(open(load_directory, "rb"))
    X, Y, schedule_array = create_new_dataset(data=data, num_schedules=num_schedules)
    for i, each_element in enumerate(X):
        X[i] = each_element + list(range(20))

    X_train = copy.deepcopy(X)
    Y_train = copy.deepcopy(Y)
    return X_train, Y_train



def test(X_train, Y_train):
    load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
        100) + 'test_dist_early_hili_naive.pkl'
    data = pickle.load(open(load_directory, "rb"))
    X, Y, schedule_array = create_new_dataset(data=data, num_schedules=100)
    for i, each_element in enumerate(X):
        X[i] = each_element + list(range(20))

    X_test = X
    Y_test = Y
    clf = DecisionTreeClassifier(max_depth=15)
    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_train)
    print(accuracy_score(Y_train, y_pred))

    y_pred_test = clf.predict(X_test)
    print(accuracy_score(Y_test, y_pred_test))

    counter = 0
    acc =0
    accs = []
    for j,i in enumerate(y_pred_test):
        at_20 = False
        if j % 20 == 0 and j != 0:
            counter += 1
            at_20 = True
        if at_20:
            accs.append(acc/20)
            acc = 0
        else:
            if i == Y_test[j]:
                acc += 1
    print(np.mean(accs))
    print(np.std(accs))




    tree.export_graphviz(clf, out_file='tree.dot')

a,b = train()
test(a,b)
