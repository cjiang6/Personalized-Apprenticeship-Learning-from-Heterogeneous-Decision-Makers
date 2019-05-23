"""
Testing the NN_small. This is expected to do much worse than the BDDT
"""

import torch
import sys
import torch.nn as nn

sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')
from scheduling_env.alpha_div import AlphaLoss
import numpy as np
from scheduling_env.argument_parser import Logger
import pickle
from torch.autograd import Variable
from utils.pairwise_utils import create_new_data, create_sets_of_20_from_x_for_pairwise_comparisions, save_performance_results

sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier




class Pairwise_DT:
    """
    class structure to train the BDT with a certain alpha.
    This class handles training the BDT, evaluating the BDT, and saving
    """

    def __init__(self):
        num_schedules = 150
        self.num_schedules = num_schedules

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + 'dist_early_hili_pairwise.pkl'

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


        data_matrix = []
        output_matrix = []

        # variables to keep track of loss and number of tasks trained over

        while True:
            # sample a timestep before the cutoff for cross_validation
            set_of_twenty = np.random.choice(self.start_of_each_set_twenty)
            truth = self.Y[set_of_twenty]

            # find feature vector of true action taken
            phi_i_num = truth + set_of_twenty
            phi_i = self.X[phi_i_num]
            phi_i_numpy = np.asarray(phi_i)

            # iterate over pairwise comparisons
            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:  # if counter == phi_i_num:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_j_numpy
                    data_matrix.append(list(feature_input))

                    output_matrix.append(1)
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_j_numpy
                    data_matrix.append(list(feature_input))

                    output_matrix.append(0)



            if len(data_matrix) > 30000:
                return data_matrix, output_matrix


    def generate_test_data(self):
        """
        Generates a bunch of counterfactual data (poorly done)
        :return:
        """

        num_schedules = 100
        # load in new data
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            num_schedules) + 'test_dist_early_hili_pairwise.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_data(num_schedules, data)

        data_matrix = []
        output_matrix = []

        # variables to keep track of loss and number of tasks trained over

        for j in range(0, num_schedules):
            # sample a timestep before the cutoff for cross_validation
            schedule_bounds = schedule_array[j]
            step = schedule_bounds[0]
            truth = Y[step]

            # find feature vector of true action taken
            phi_i_num = truth + step
            while step < schedule_bounds[1]:
                for counter in range(step, step + 20):
                    if counter == phi_i_num:  # if counter == phi_i_num:
                        phi_j = X[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_j_numpy
                        data_matrix.append(list(feature_input))

                        output_matrix.append(1)
                    else:
                        phi_j = X[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_j_numpy
                        data_matrix.append(list(feature_input))

                        output_matrix.append(0)

                        # add average loss to array
                step += 20

        return data_matrix, output_matrix


    def evaluate(self,clf):

        """
        Evaluate performance of a DT
        :return:
        """


        num_schedules = 100
        # load in new data
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            num_schedules) + 'test_dist_early_hili_pairwise.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_data(num_schedules, data)

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        for j in range(0, num_schedules):
            schedule_bounds = schedule_array[j]
            step = schedule_bounds[0]
            while step < schedule_bounds[1]:
                probability_vector = np.zeros((1, 20))

                for m, counter in enumerate(range(step, step + 20)):
                    phi_i = X[counter]
                    phi_i_numpy = np.asarray(phi_i)

                    feature_input = phi_i_numpy


                    # push through nets
                    preference_prob = clf.predict(feature_input.reshape(1,-1))
                    probability_vector[0][m] = preference_prob[0]
                # feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13))

                # Set of twenty is completed
                highest_val = max(probability_vector[0])
                all_indexes_that_have_highest_val = [i for i, e in enumerate(list(probability_vector[0])) if e == highest_val]
                # top 1
                choice = np.random.choice(all_indexes_that_have_highest_val)
                # choice = np.argmax(probability_vector)

                # top 1
                # choice = np.argmax(column_vec)


                # Then do training update loop
                truth = Y[step]

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
        print(np.std(percentage_accuracy_top1))


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


