"""
Testing the NN_small. This is expected to do much worse than the BDDT
"""

import torch
import sys
import torch.nn as nn
import itertools

# sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')
import numpy as np
from scheduling_env.argument_parser import Logger
import pickle
from torch.autograd import Variable
from utils.pairwise_utils import create_new_data, find_which_schedule_this_belongs_to, create_sets_of_20_from_x_for_pairwise_comparisions, save_performance_results
from sklearn.cluster import KMeans
from scheduling_env.generate_results_of_hypothesis.pairwise.train_autoencoder import Autoencoder, AutoEncoderTrain
from utils.naive_utils import create_new_dataset

sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)
np.random.seed(50)


class NNSmall(nn.Module):
    """
    number of parameters should be it is N*(|X| + |E| + |C|) + |Y|*L
    where N=num_nodes, X=input_sample, E=bayesian_embedding, Y=output_classes, L=num_leaves, C=comparator_vector (1 for non_vec, |X| for vec)
    I will consider a baseline case of 64 nodes, 8 is the size of bayesian embedding, 13 is the size of the input, L is 6, Y is 2. Comparators is 13.
    # updated: num_nodes*(3*input_size + 3*embedding_size) + num_leaves*output_size,
    so N*(|X| + |E| + |C| + |S|) + Y*L where N=num_nodes, X=input_sample, E=bayesian_embedding, C=comparator_vector, S=selector_vector, Y=output_classes, L=num_leaves
    but |C| = |S| = (|X|+|E|)
    so it's N*(3*(|X|+|E|))

    In total this is 2188, what was returned was 4106.
    This one has 3649 (little smaller but I think its close enough) # TODO: maybe add one more layer
    NOTE: this line returns number of model params  pytorch_total_params = sum(p.numel() for p in self.model.parameters()),
    NOTE: only trainable params is pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    """

    def __init__(self):
        super(NNSmall, self).__init__()
        self.fc1 = nn.Linear(13, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(32, 32)
        self.relu21 = nn.ReLU()
        self.fc22 = nn.Linear(32, 32)
        self.relu22 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        forward pass
        :param x: i_minus_j or vice versa
        :return:
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc21(x)
        x = self.relu21(x)
        x = self.fc22(x)
        x = self.relu22(x)
        x = self.fc3(x)
        x = self.sig(x)

        return x


class NNTrain:
    """
    class structure to train the NN for a certain amount of schedules.
    This class handles training the NN, evaluating the NN, and saving the results
    """

    def __init__(self):
        self.arguments = Logger()
        self.alpha = .9
        self.num_schedules = 150
        self.home_dir = self.arguments.home_dir

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + 'dist_early_hili_pairwise.pkl'

        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_data(self.num_schedules, self.data)
        self.start_of_each_set_twenty = create_sets_of_20_from_x_for_pairwise_comparisions(self.X)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model1 = NNSmall().to(device)
        model2 = NNSmall().to(device)
        model3 = NNSmall().to(device)

        self.models = [model1, model2, model3]

        opt1 = torch.optim.Adam(model1.parameters())
        opt2 = torch.optim.Adam(model2.parameters())
        opt3 = torch.optim.Adam(model3.parameters())

        self.optimizers = [opt1, opt2, opt3]
        schedule_matrix_load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/' + str(self.num_schedules) + 'matrixes.pkl'
        self.matrices = pickle.load(open(schedule_matrix_load_directory, "rb"))

        self.kmeans_model, self.label = self.cluster_matrices(self.matrices, self.num_schedules)

    @staticmethod
    def cluster_matrices(matrices, num_schedules):
        # vectorize each matrix
        vectorized_set = []
        for i in matrices:
            vectorized = i.reshape(20 * 2048, 1)
            vectorized_set.append(vectorized)
        kmeans = KMeans(n_clusters=3, random_state=0)  # random state makes it deterministic
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
        criterion = torch.nn.BCELoss()

        # variables to keep track of loss and number of tasks trained over

        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            set_of_twenty = np.random.choice(self.start_of_each_set_twenty)
            truth = self.Y[set_of_twenty]
            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array, set_of_twenty)
            cluster_num = self.label[which_schedule]

            # find feature vector of true action taken
            phi_i_num = truth + set_of_twenty
            phi_i = self.X[phi_i_num]
            phi_i_numpy = np.asarray(phi_i)
            running_loss_predict_tasks = 0
            num_iterations_predict_task = 0
            # iterate over pairwise comparisons
            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:  # if counter == phi_i_num:
                    label = torch.ones((1, 1))
                else:
                    label = torch.zeros((1, 1))
                phi_j = self.X[counter]
                phi = np.asarray(phi_j)
                feature_input = phi

                if torch.cuda.is_available():
                    feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                    label = Variable(torch.Tensor(label).cuda())
                else:
                    feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                    label = Variable(torch.Tensor(label.reshape(1, 1)))

                output = self.models[cluster_num].forward(feature_input)

                self.optimizers[cluster_num].zero_grad()
                loss = criterion(output, label)
                if counter == phi_i_num:
                    loss *= 25

                if torch.isnan(loss):
                    print('nan occurred at iteration ', total_iterations, ' at', num_iterations_predict_task)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizers[cluster_num].step()
                running_loss_predict_tasks += loss.item()
                num_iterations_predict_task += 1

            total_loss_array.append(running_loss_predict_tasks / num_iterations_predict_task)

            total_iterations += 1

            if total_iterations % 50 == 49:
                print('total loss (average for each 40, averaged) at iteration ', total_iterations, ' is ', np.mean(total_loss_array[-40:]))

            if total_iterations > 15000 and np.mean(total_loss_array[-100:]) - np.mean(total_loss_array[-500:]) < convergence_epsilon:
                training_done = True

    def create_iterables(self):
        """
        adds all possible state combinations
        :return:
        """
        iterables = [[0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1]]
        states = []
        for t in itertools.product(*iterables):
            states.append(t)
        return states

    def pass_in_embedding_out_state_ID(self, states, binary):
        """
        pass in a binary embedding, and itll return the state id
        :param binary:
        :return:
        """
        binary_as_tuple = tuple(binary)
        index = states.index(binary_as_tuple)
        return index

    # noinspection PyArgumentList

    def evaluate_on_test_data(self, models, schedules_trained_on, load_in_model=False):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        Note this function is called after training convergence
        :return:
        """

        autoencoder_class = AutoEncoderTrain(150)
        checkpoint = torch.load('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/Autoencoder150.tar')
        autoencoder_class.model.load_state_dict(checkpoint['nn_state_dict'])
        states = self.create_iterables()

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            100) + 'test_dist_early_hili_naive.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X_naive, Y_naive, schedule_array = create_new_dataset(data, 100)
        for i, each_element in enumerate(X_naive):
            X_naive[i] = each_element + list(range(20))

        num_schedules = 100
        # load in new data
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            num_schedules) + 'test_dist_early_hili_pairwise.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_data(num_schedules, data)

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []
        mean_input = [1.3277743, 0.32837677, 1.4974482, -1.3519306, -0.64621973, 0.10534518, -2.338118, -2.7345326, 1.7558736, -3.0746384, -3.485554]

        for j in range(0, num_schedules):
            current_schedule_matrix = np.zeros((2048, 20))
            schedule_bounds = schedule_array[j]
            step = schedule_bounds[0]
            while step < schedule_bounds[1]:
                probability_vector = np.zeros((1, 20))
                if current_schedule_matrix.sum() == 0:
                    cluster_num = self.kmeans_model.predict(current_schedule_matrix.reshape(1, -1))
                else:
                    matrix = np.divide(current_schedule_matrix, current_schedule_matrix.sum())
                    cluster_num = self.kmeans_model.predict(matrix.reshape(1, -1))

                for m, counter in enumerate(range(step, step + 20)):
                    phi_i = X[counter]
                    phi_i_numpy = np.asarray(phi_i)

                    feature_input = phi_i_numpy

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))

                    # push through nets
                    preference_prob = models[int(cluster_num)].forward(feature_input)
                    probability_vector[0][m] = preference_prob[0].data.detach()[
                        0].item()  # TODO: you can do a check if only this line leads to the same thing as the line below
                    # probability_matrix[n][m] = preference_prob[0].data.detach()[1].item()

                print(probability_vector)
                highest_val = max(probability_vector[0])
                all_indexes_that_have_highest_val = [i for i, e in enumerate(list(probability_vector[0])) if e == highest_val]
                # top 1
                choice = np.random.choice(all_indexes_that_have_highest_val)
                # choice = np.argmax(probability_vector)

                # top 3
                _, top_three = torch.topk(torch.Tensor(probability_vector), 3)

                # Then do training update loop
                truth = Y[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1

                # index top 3
                if truth in top_three:
                    prediction_accuracy[1] += 1

                embedding_copy = np.zeros((1, 11))
                input_element = autoencoder_class.model.forward_only_encoding(Variable(torch.Tensor(np.asarray(X_naive[int(step / 20)]).reshape(1, 242)).cuda()))
                for z, each_element in enumerate(mean_input):
                    if each_element > input_element[0][z].item():
                        embedding_copy[0][z] = 0
                    else:
                        embedding_copy[0][z] = 1
                index = self.pass_in_embedding_out_state_ID(states, embedding_copy[0])
                action = Y[step]
                current_schedule_matrix[index][int(action)] += 1
                # add average loss to array
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'pointwise_NN_kmeans.pkl')

        return percentage_accuracy_top1

    def save_trained_nets(self, name):
        """
        saves the model
        :return:
        """
        torch.save({'nn_state_dict': self.models,
                    'parameters': self.arguments},
                   '/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/NN_' + name + '.tar')


def main():
    """
    entry point for file
    :return:
    """

    num_schedules = 150
    trainer = NNTrain()
    trainer.train()
    per_schedule_test_accs = trainer.evaluate_on_test_data(trainer.models, num_schedules)
    file = open('scheduling_env_results.txt', 'a')
    file.write('k_means: mean: ' +
               str(np.mean(per_schedule_test_accs)) +
               ', std: ' + str(np.std(per_schedule_test_accs)) +
               '\n')
    file.close()


if __name__ == '__main__':
    main()
