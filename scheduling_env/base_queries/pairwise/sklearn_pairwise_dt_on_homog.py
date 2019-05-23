"""
create a pairwise dt that should achieve .99% accuracy on training
"""

import sys

# sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')

import numpy as np
import pickle

from utils.pairwise_utils import create_new_data

sys.path.insert(0, '../')

from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# TODO: add modular learning rate
# TODO: check how important batching is, maybe parallelism is better?
# TODO: import functions from pairwise_utils


class Pairwise_DT:
    """
    class structure to train the BDT with a certain alpha.
    This class handles training the BDT, evaluating the BDT, and saving
    """

    def __init__(self):
        num_schedules = 150
        self.num_schedules = num_schedules

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + '_homog_deadline_pairwise.pkl'

        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_data(num_schedules, self.data)
        self.start_of_each_set_twenty = self.create_sets_of_20_from_x_for_pairwise_comparisions()

    def create_sets_of_20_from_x_for_pairwise_comparisions(self):
        """
        Create sets of 20 to denote each timestep for all schedules
        :return: range(0, length_of_X, 20)
        """
        length_of_X = len(self.X)
        return list(range(0, length_of_X, 20))

    def find_which_schedule_this_belongs_to(self, sample_val):
        """
        Takes a sample and determines with schedule this belongs to.
        Note: A schedule is task * task sized
        :param sample_val: an int
        :return: schedule num
        """
        for i, each_array in enumerate(self.schedule_array):
            if each_array[0] <= sample_val <= each_array[1]:
                return i
            else:
                continue

    def generate_data(self):
        """
        Generates a bunch of counterfactual data (poorly done)
        :return:
        """

        cv_cutoff = int(.8 * len(self.start_of_each_set_twenty))

        data_matrix = []
        output_matrix = []

        # variables to keep track of loss and number of tasks trained over

        while True:
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(cv_cutoff)
            set_of_twenty = self.start_of_each_set_twenty[rand_timestep_within_sched]
            truth = self.Y[set_of_twenty]

            # find feature vector of true action taken
            phi_i_num = truth + set_of_twenty
            phi_i = self.X[phi_i_num]
            phi_i_numpy = np.asarray(phi_i)

            # iterate over pairwise comparisons
            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:  # if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_i_numpy - phi_j_numpy
                    data_matrix.append(list(feature_input))

                    output_matrix.append(1)

            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_j_numpy - phi_i_numpy

                    data_matrix.append(list(feature_input))
                    output_matrix.append(0)

            if len(data_matrix) > 300000:
                return data_matrix, output_matrix


    def generate_test_data(self):
        """
        Generates a bunch of counterfactual data (poorly done)
        :return:
        """

        cv_cutoff = int(.8 * len(self.start_of_each_set_twenty))

        data_matrix = []
        output_matrix = []

        # variables to keep track of loss and number of tasks trained over

        while True:
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(cv_cutoff, len(self.start_of_each_set_twenty))
            set_of_twenty = self.start_of_each_set_twenty[rand_timestep_within_sched]
            truth = self.Y[set_of_twenty]

            # find feature vector of true action taken
            phi_i_num = truth + set_of_twenty
            phi_i = self.X[phi_i_num]
            phi_i_numpy = np.asarray(phi_i)

            # iterate over pairwise comparisons
            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:  # if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_i_numpy - phi_j_numpy
                    data_matrix.append(list(feature_input))

                    output_matrix.append(1)

            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_j_numpy - phi_i_numpy

                    data_matrix.append(list(feature_input))
                    output_matrix.append(0)

            if len(data_matrix) > 300000:
                return data_matrix, output_matrix


    def evaluate(self,clf):

        """
        Evaluate performance of a DT
        :return:
        """


        prediction_accuracy = [0]
        percentage_accuracy_top1 = []



        # for rest of schedule
        # i = .8 * len(self.start_of_each_set_twenty)
        # num_test_schedules = 150 * .2
        for j in range(int(0), int(self.num_schedules*.8)): # training

            schedule_bounds = self.schedule_array[j]
            step = schedule_bounds[0]
            while step < schedule_bounds[1]:
                probability_matrix = np.zeros((20, 20))

                for m, counter in enumerate(range(step, step + 20)):
                    phi_i = self.X[counter]
                    phi_i_numpy = np.asarray(phi_i)

                    # for each set of twenty
                    for n, second_counter in enumerate(range(step, step + 20)):
                        # fill entire array with diagnols set to zero
                        if second_counter == counter:  # same as m = n
                            continue
                        phi_j = self.X[second_counter]
                        phi_j_numpy = np.asarray(phi_j)

                        feature_input = phi_i_numpy - phi_j_numpy


                        # push through nets
                        preference_prob = clf.predict(feature_input.reshape(1,-1))
                        probability_matrix[m][n] = preference_prob


                # Set of twenty is completed
                column_vec = np.sum(probability_matrix, axis=1)

                # top 1
                choice = np.argmax(column_vec)


                # Then do training update loop
                truth = self.Y[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1


                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20)
            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)


            prediction_accuracy = [0]

        print(np.mean(percentage_accuracy_top1))


def main():
    """
    entry point for file
    :return:
    """
    trainer = Pairwise_DT()
    X, Y = trainer.generate_data()
    clf = DecisionTreeClassifier(max_depth=20)
    clf.fit(X, Y)

    y_pred = clf.predict(X)
    print(accuracy_score(Y, y_pred))

    trainer.evaluate(clf)

    X_test, Y_test = trainer.generate_test_data()
    y_pred_test = clf.predict(X_test)
    print(accuracy_score(Y_test, y_pred_test))

    tree.export_graphviz(clf, out_file='tree_pairwise.dot')


if __name__ == '__main__':
    main()
