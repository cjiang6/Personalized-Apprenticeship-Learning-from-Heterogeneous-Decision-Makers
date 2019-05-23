"""
k-means NN
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
from utils.global_utils import save_pickle
from sklearn.cluster import KMeans
from utils.pairwise_utils import create_new_data, create_sets_of_20_from_x_for_pairwise_comparisions, find_which_schedule_this_belongs_to, save_performance_results
from scheduling_env.generate_results_of_hypothesis.pairwise.nn_small_pairwise import NNSmall
from utils.naive_utils import create_new_dataset
from scheduling_env.generate_results_of_hypothesis.pairwise.train_autoencoder import Autoencoder
import itertools
sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)




class KMeansNNTrain():
    def __init__(self, num_schedules):
        self.arguments = Logger()
        self.alpha = .9
        self.num_schedules = num_schedules
        self.home_dir = self.arguments.home_dir

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + 'high_low_hetero_deadline_pairwise.pkl'

        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_data(self.num_schedules, self.data)
        self.start_of_each_set_twenty = create_sets_of_20_from_x_for_pairwise_comparisions(self.X)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.models = [NNSmall().to(device), NNSmall().to(device), NNSmall().to(device)]

        self.opts = [torch.optim.SGD(self.models[0].parameters(), lr=.0001, weight_decay=.1),
                     torch.optim.SGD(self.models[1].parameters(), lr=.0001, weight_decay=.1),
                     torch.optim.SGD(self.models[2].parameters(), lr=.0001, weight_decay=.1)]



        schedule_matrix_load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/' + str(self.num_schedules) + 'matrices.pkl'
        self.matrices = pickle.load(open(schedule_matrix_load_directory, "rb"))

        self.kmeans_model, self.label = self.cluster_matrices(self.matrices, self.num_schedules)


    @staticmethod
    def cluster_matrices(matrices, num_schedules):
        # vectorize each matrix
        vectorized_set = []
        for i in matrices:
            vectorized = i.reshape(20 * 2048, 1)
            vectorized_set.append(vectorized)
        kmeans = KMeans(n_clusters=3, random_state=0) # random state makes it deterministic
        # Fitting the input data
        new_set = np.hstack(tuple(vectorized_set)).reshape(num_schedules, 20 * 2048)
        kmeans_model = kmeans.fit(np.asarray(new_set))
        labels = kmeans_model.predict(np.asarray(new_set))
        return kmeans_model, labels


    # noinspection PyArgumentList
    def train(self):
        """
        Trains NN.
        Randomly samples a schedule and timestep within that schedule, produces training data using x_i - x_j
        and trains upon that.
        :return:
        """

        total_iterations = 0
        convergence_epsilon = .01
        when_to_save = 1000
        distribution_epsilon = .0001
        training_done = False
        total_loss_array = []

        loss_func = AlphaLoss(.9)

        # variables to keep track of loss and number of tasks trained over

        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            set_of_twenty = np.random.choice(self.start_of_each_set_twenty)
            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array, set_of_twenty)

            # get actual task scheduled
            truth = self.Y[set_of_twenty]

            # choose cluster based on value produced by kmeans
            cluster_num = self.label[which_schedule]


            network_to_train = self.models[cluster_num]
            optimizer_for_net = self.opts[cluster_num]
            # find feature vector of true action taken
            phi_i_num = truth + set_of_twenty
            phi_i = self.X[phi_i_num]
            phi_i_numpy = np.asarray(phi_i)
            running_loss_predict_tasks = 0
            num_iterations_predict_task = 0
            # iterate over pairwise comparisons
            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:  # if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_i_numpy - phi_j_numpy

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        P = Variable(torch.Tensor([1 - distribution_epsilon, distribution_epsilon]).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        P = Variable(torch.Tensor([1 - distribution_epsilon, distribution_epsilon]))

                    output = network_to_train.forward(feature_input)

                    if torch.isnan(output[0][0]).item() == 1:
                        print('hi')
                    optimizer_for_net.zero_grad()
                    loss = loss_func.forward(P, output)

                    if torch.isnan(loss):
                        print(self.alpha, ' :nan occurred at iteration ', total_iterations, ' at', num_iterations_predict_task)

                    if loss.item() < .001 or loss.item() > 50:
                        pass
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(network_to_train.parameters(), 0.5)
                        optimizer_for_net.step()
                    running_loss_predict_tasks += loss.item()
                    num_iterations_predict_task += 1

            # second loop
            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_j_numpy - phi_i_numpy

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        P = Variable(torch.Tensor([distribution_epsilon, 1 - distribution_epsilon]).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        P = Variable(torch.Tensor([distribution_epsilon, 1 - distribution_epsilon]))

                    output = network_to_train.forward(feature_input)
                    if torch.isnan(output[0][0]).item() == 1:
                        print('hi')
                    optimizer_for_net.zero_grad()
                    loss = loss_func.forward(P, output)

                    if loss.item() < .001 or loss.item() > 50:
                        pass
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(network_to_train.parameters(), 0.5)
                        optimizer_for_net.step()

                    running_loss_predict_tasks += loss.item()

                    num_iterations_predict_task += 1

            total_loss_array.append(running_loss_predict_tasks / num_iterations_predict_task)

            total_iterations += 1

            if total_iterations % 50 == 49:
                print('total loss (average for each 40, averaged) at iteration ', total_iterations, ' is ', np.mean(total_loss_array[-40:]))

            if total_iterations % when_to_save == when_to_save - 1:
                self.save_trained_nets('kmeans_nn_small' + str(self.num_schedules))

            if total_iterations > 5000 and np.mean(total_loss_array[-100:]) - np.mean(total_loss_array[-500:]) < convergence_epsilon:
                training_done = True

    # noinspection PyArgumentList
    def evaluate_on_test_data(self, models, schedules_trained_on):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        Note this function is called after training convergence
        :return:
        """
        num_schedules = 75
        # load in new data
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            num_schedules) + 'test_high_low_hetero_deadline_pairwise.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_data(num_schedules, data)
        
        ### take a side step and do some of the clustering stuff
        autoencoder_class = AutoEncoderTrain(num_schedules)
        autoencoder_class.model.load('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/Autoencoder' + str(schedules_trained_on) + '.tar')
        autoencoder_class.compute_mean()
        autoencoder_class.create_iterables()

        autoencoder_class.round_each_encoding_and_create_array()
        autoencoder_class.populate_a_matrix_per_schedule()
        test_matrices = autoencoder_class.save_matrices()
        
        kmeans_model, labels = self.cluster_matrices(test_matrices, num_schedules)



        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        for j in range(0, num_schedules):
            schedule_bounds = schedule_array[j]
            step = schedule_bounds[0]
            network_to_train = models[labels[j]]

            while step < schedule_bounds[1]:
                probability_matrix = np.zeros((20, 20))

                for m, counter in enumerate(range(step, step + 20)):
                    phi_i = X[counter]
                    phi_i_numpy = np.asarray(phi_i)

                    # for each set of twenty
                    for n, second_counter in enumerate(range(step, step + 20)):
                        # fill entire array with diagnols set to zero
                        if second_counter == counter:  # same as m = n
                            continue
                        phi_j = X[second_counter]
                        phi_j_numpy = np.asarray(phi_j)

                        feature_input = phi_i_numpy - phi_j_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())

                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))

                        # push through nets
                        preference_prob = network_to_train.forward(feature_input)
                        probability_matrix[m][n] = preference_prob[0].data.detach()[
                            0].item()  # TODO: you can do a check if only this line leads to the same thing as the line below
                        # probability_matrix[n][m] = preference_prob[0].data.detach()[1].item()

                # Set of twenty is completed
                column_vec = np.sum(probability_matrix, axis=1)

                # top 1
                choice = np.argmax(column_vec)

                # top 3
                _, top_three = torch.topk(torch.Tensor(column_vec), 3)

                # Then do training update loop
                truth = Y[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1

                # index top 3
                if truth in top_three:
                    prediction_accuracy[1] += 1

                # add average loss to array
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'HIFI_LIFI_k_means_nn_small_pairwise' + str(schedules_trained_on) + '.pkl')
        
    def save_trained_nets(self, name):
        """
        saves the model
        :return:
        """
        torch.save({'nn_1_state_dict': self.models[0].state_dict(),
                    'nn_2_state_dict': self.models[1].state_dict(),
                    'nn_3_state_dict': self.models[2].state_dict()},
                   '/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/k_means_nn_' + name + '.tar')


# can also be used for gmm
class AutoEncoderTrain:
    """
    create and train the autoencoder
    """

    def __init__(self):

        self.num_schedules = 75
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + 'test_high_low_hetero_deadline_pairwise.pkl'
        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_dataset(num_schedules=self.num_schedules, data=self.data)
        for i, each_element in enumerate(self.X):
            self.X[i] = each_element + list(range(20))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder().to(device)

        print(self.model.state_dict())
        self.opt = torch.optim.SGD(self.model.parameters(), lr=.0001)
        self.mean_embedding = None
        self.embedding_np = None
        self.matrices = None
        self.total_binary_embeddings = None
        self.states = None

    # noinspection PyArgumentList
    def compute_mean(self):
        """
        computes the mean embedding by first computing all embeddings for every step of the schedule,
        adding them to a numpy array and computing the avg
        :return:
        """
        # load_in_all_parameters(self.save_directory, self.auto_encoder)
        for i, data_row in enumerate(self.X):
            input_nn = data_row
            if torch.cuda.is_available():
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())
            else:
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))

            prediction_embedding = self.model.forward_only_encoding(input_nn)
            print(prediction_embedding)
            if i == 0:
                self.embedding_np = prediction_embedding.data.clone().cpu().numpy()[0]
            else:
                self.embedding_np = np.vstack((self.embedding_np, prediction_embedding.data.clone().cpu().numpy()[0]))
        self.mean_embedding = np.average(self.embedding_np, axis=0)
        print('mean embedding is ', self.mean_embedding)

    def create_iterables(self):
        """
        adds all possible state combinations
        :return:
        """
        iterables = [[0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1]]
        self.states = []
        for t in itertools.product(*iterables):
            self.states.append(t)

    # noinspection PyArgumentList
    def round_each_encoding_and_create_array(self):
        """
        rounds each encoding by comparing it to the mean, and then stacks these in an array
        :return:
        """
        self.total_binary_embeddings = np.zeros((0))
        for counter, data_row in enumerate(self.X):
            input_nn = data_row
            if torch.cuda.is_available():
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())
            else:
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))

            prediction_embedding = self.model.forward_only_encoding(input_nn)

            embedding_copy = np.zeros((1, 11))

            for i, each_element in enumerate(self.mean_embedding):
                if each_element > prediction_embedding.data[0][i].item():
                    embedding_copy[0][i] = 0
                else:
                    embedding_copy[0][i] = 1

            if counter == 0:
                self.total_binary_embeddings = embedding_copy
            else:
                self.total_binary_embeddings = np.vstack((self.total_binary_embeddings, embedding_copy))

            # This should generate n schedules of binary data
        print('finished turning all elements of schedule into binary')

    def pass_in_embedding_out_state_ID(self, binary):
        """
        pass in a binary embedding, and itll return the state id
        :param binary:
        :return:
        """
        binary_as_tuple = tuple(binary)
        index = self.states.index(binary_as_tuple)
        return index

    def populate_a_matrix_per_schedule(self):
        """
        creates matrices bases on the binary embeddings
        :return:
        """
        self.matrices = []
        for i in range(self.num_schedules):
            m = np.zeros((2048, 20))
            self.matrices.append(m)
        for i, each_matrix in enumerate(self.matrices):
            # lets look at elements of schedule 1
            for j in range(self.schedule_array[i][0], self.schedule_array[i][1] + 1):
                binary_embedding = self.total_binary_embeddings[j]
                index = self.pass_in_embedding_out_state_ID(binary_embedding)
                # action taken at this instance
                action = self.Y[j]
                each_matrix[index][action] += 1
            total_sum = each_matrix.sum()
            self.matrices[i] = np.divide(each_matrix, total_sum)

        print('n matrices have been generated')

    # def cluster_matrices(self):
    #     # vectorize each matrix
    #     vectorized_set = []
    #     for i in self.matrices:
    #         vectorized = i.reshape(20 * 2048, 1)
    #         vectorized_set.append(vectorized)
    #     kmeans = KMeans(n_clusters=3)
    #     # Fitting the input data
    #     new_set = np.hstack(tuple(vectorized_set)).reshape(self.num_schedules, 20 * 2048)
    #     self.kmeans = kmeans.fit(np.asarray(new_set))
    #     self.label = self.kmeans.predict(np.asarray(new_set))

    def save_matrices(self):
        """
        saves the matrices so these can be used to cluster in the gmm etc.
        :return:
        """
        save_pickle('/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/', self.matrices, str(self.num_schedules) + 'test_matrices.pkl')
        return self.matrices


def main():
    """
    entry point for file
    :return:
    """
    for num_schedules in (3, 9, 15, 150, 1500):
        trainer = KMeansNNTrain(num_schedules)
        trainer.train()
        trainer.evaluate_on_test_data(trainer.models, num_schedules)


if __name__ == '__main__':
    main()
    
    
    
    


    
    
    
