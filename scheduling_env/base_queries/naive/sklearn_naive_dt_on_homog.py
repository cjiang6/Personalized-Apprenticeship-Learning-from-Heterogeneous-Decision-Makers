"""
create a dt that should achieve .99% accuracy on training
"""

import sys

import pickle
from sklearn.tree import DecisionTreeClassifier
from utils.naive_utils import create_new_dataset
from sklearn.metrics import accuracy_score
from sklearn import tree

sys.path.insert(0, '../')

# Naive


num_schedules = 150
total_loss_array = []

load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
    num_schedules) + '_task_num_homog_deadline_naive.pkl'

data = pickle.load(open(load_directory, "rb"))
X, Y, schedule_array = create_new_dataset(data=data, num_schedules=num_schedules)
for i, each_element in enumerate(X):
    X[i] = each_element + list(range(20))

X_train = X[0:int(len(X) * .8)]
Y_train = Y[0:int(len(X) * .8)]
X_test = X[int(len(X) * .8):int(len(X))]
Y_test = Y[int(len(X) * .8):int(len(X))]
clf = DecisionTreeClassifier(max_depth=15)
clf.fit(X_train, Y_train)

y_pred = clf.predict(X_train)
print(accuracy_score(Y_train, y_pred))

y_pred_test = clf.predict(X_test)
print(accuracy_score(Y_test, y_pred_test))

tree.export_graphviz(clf, out_file='tree.dot')
