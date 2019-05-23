"""
Testing the NN_small. This is expected to do much worse than the BDDT
"""

import torch
import sys
import torch.nn as nn

# sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')
from scheduling_env.alpha_div import AlphaLoss
import numpy as np
from scheduling_env.argument_parser import Logger
import pickle
from torch.autograd import Variable
from utils.pairwise_utils import create_new_data, create_sets_of_20_from_x_for_pairwise_comparisions, save_performance_results, find_which_schedule_this_belongs_to
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier



sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)
class EmbeddingModule(nn.Module):
    """
    embedding class (allows us to access parameters directly)
    """

    def __init__(self):
        super(EmbeddingModule, self).__init__()
        self.embedding = nn.Parameter(torch.randn(1, 3))

    def forward(self):
        """
        doesn't do anything
        :return:
        """
        return

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
        self.fc1 = nn.Linear(16, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(32, 32)
        self.relu21 = nn.ReLU()
        self.fc22 = nn.Linear(32, 32)
        self.relu22 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()
        self.EmbeddingList = nn.ModuleList(EmbeddingModule() for _ in range(1))

    def forward(self, x):
        """
        forward pass
        :param x: i_minus_j or vice versa
        :return:
        """
        w = self.EmbeddingList[0].embedding
        x = torch.cat([x, w], dim=1)
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

    def set_bayesian_embedding(self, embedding):
        """
        sets embedding into BNN
        :param embedding:
        :return:
        """
        for n, i in enumerate(embedding.reshape(3)):
            self.EmbeddingList[0].embedding.data[0][n].fill_(i)

    def get_bayesian_embedding(self):
        """
        gets embedding inside BNN
        :return:
        """
        return self.EmbeddingList[0].embedding

class NNTrain:
    """
    class structure to train the NN for a certain amount of schedules.
    This class handles training the NN, evaluating the NN, and saving the results
    """

    def __init__(self, num_schedules):
        self.arguments = Logger()
        self.alpha = .9
        self.num_schedules = num_schedules
        self.home_dir = self.arguments.home_dir

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + 'dist_early_hili_pairwise.pkl'

        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_data(self.num_schedules, self.data)
        self.start_of_each_set_twenty = create_sets_of_20_from_x_for_pairwise_comparisions(self.X)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = NNSmall().to(device)

        print(self.model.state_dict())
        # set up main optimizer to optimize model parameters and embedding parameters with different learning rates
        self.opt = torch.optim.Adam([{'params': list(self.model.parameters())[:-1]}, {'params': self.model.EmbeddingList.parameters(), 'lr': .01}], lr=.0001)
        self.embedding_optimizer = torch.optim.SGD(self.model.EmbeddingList.parameters(), lr=.1)
        self.embedding_list = [torch.ones(3) * 1 / 3 for _ in range(self.num_schedules)]


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
            # choose a random timestep within any schedule
            set_of_twenty = np.random.choice(self.start_of_each_set_twenty)

            # get the true action scheduled
            truth = self.Y[set_of_twenty]
            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array, set_of_twenty)

            # set the embedding
            self.model.set_bayesian_embedding(self.embedding_list[which_schedule])

            # find feature vector of true action taken
            phi_i_num = truth + set_of_twenty
            phi_i = self.X[phi_i_num]
            phi_i_numpy = np.asarray(phi_i)
            running_loss_predict_tasks = 0
            num_iterations_predict_task = 0
            # iterate over pairwise comparisons
            for counter in range(set_of_twenty, set_of_twenty + 20):
                # positive counterfactuals
                if counter == phi_i_num:  # if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_i_numpy - phi_j_numpy

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        label = Variable(torch.Tensor(torch.ones((1,1))).cuda())

                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        label = Variable(torch.Tensor(torch.ones((1, 1))))

                    output = self.model.forward(feature_input)


                    self.opt.zero_grad()
                    loss = criterion(output, label)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.opt.step()
                    running_loss_predict_tasks += loss.item()
                    num_iterations_predict_task += 1

            # Negative counterfactuals
            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_j_numpy - phi_i_numpy

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        label = Variable(torch.Tensor(torch.zeros((1,1))).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        label = Variable(torch.Tensor(torch.zeros((1,1))))

                    output = self.model.forward(feature_input)
                    loss = criterion(output, label)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.opt.step()

                    running_loss_predict_tasks += loss.item()

                    num_iterations_predict_task += 1


            self.embedding_list[which_schedule] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy()[0])  # very ugly
            print(self.embedding_list[which_schedule])
            total_loss_array.append(running_loss_predict_tasks / num_iterations_predict_task)

            total_iterations += 1

            if total_iterations % 50 == 49:
                print('total loss (average for each 40, averaged) at iteration ', total_iterations, ' is ', np.mean(total_loss_array[-40:]))

            if total_iterations % when_to_save == when_to_save - 1:
                self.save_trained_nets('nn_small' + str(self.num_schedules))

            if total_iterations > 2000 and np.mean(total_loss_array[-100:]) - np.mean(total_loss_array[-500:]) < convergence_epsilon:
                training_done = True
                print(self.model.state_dict())

    # noinspection PyArgumentList

    def evaluate_on_test_data(self):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        Note this function is called after training convergence
        :return:
        """
        num_schedules = 100
        # load in new data
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            num_schedules) + 'test_dist_early_hili_pairwise.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_data(num_schedules, data)
        # define embedding things
        embedding_list = [torch.ones(3) * 1 / 3 for i in range(num_schedules)]
        # learning rate increased
        self.opt = torch.optim.SGD(self.model.EmbeddingList.parameters(), lr=.0001)

        criterion = torch.nn.BCELoss()

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        for j in range(0, num_schedules):
            schedule_bounds = schedule_array[j]
            step = schedule_bounds[0]
            self.model.set_bayesian_embedding(embedding_list[j])
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

                        # push through nets to get preferences
                        preference_prob = self.model.forward(feature_input)
                        probability_matrix[m][n] = preference_prob[0].data.detach()[
                            0].item()  # TODO: you can do a check if only this line leads to the same thing as the line below
                        # probability_matrix[n][m] = preference_prob[0].data.detach()[1].item()

                # Set of twenty is completed
                column_vec = np.sum(probability_matrix, axis=1)

                # top 1
                highest_val = max(column_vec)
                all_indexes_that_have_highest_val = [i for i, e in enumerate(list(column_vec)) if e == highest_val]
                if len(all_indexes_that_have_highest_val) > 1:
                    print('length of indexes greater than 1: ', all_indexes_that_have_highest_val)
                # top 1
                choice = np.random.choice(all_indexes_that_have_highest_val)

                # top 3
                _, top_three = torch.topk(torch.Tensor(column_vec), 3)


                truth = Y[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1

                # index top 3
                if truth in top_three:
                    prediction_accuracy[1] += 1

                # Then do training update loop

                phi_i_num = truth + step
                phi_i = X[phi_i_num]
                phi_i_numpy = np.asarray(phi_i)
                # iterate over pairwise comparisons
                for counter in range(step, step + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = X[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_i_numpy - phi_j_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            label = Variable(torch.Tensor(torch.ones((1, 1))).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            label = Variable(torch.Tensor(torch.ones((1, 1))))

                        output = self.model(feature_input)
                        loss = criterion(output, label)
                        # prepare optimizer, compute gradient, update params

                        self.embedding_optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.embedding_optimizer.step()
                        print(self.model.EmbeddingList.state_dict())

                for counter in range(step, step + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = X[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_j_numpy - phi_i_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            label = Variable(torch.Tensor(torch.zeros((1, 1))).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            label = Variable(torch.Tensor(torch.zeros((1, 1))))

                        output = self.model.forward(feature_input)

                        self.embedding_optimizer.zero_grad()
                        loss = criterion(output, label)

                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.embedding_optimizer.step()


                # add average loss to array
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)
            embedding_list[j] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy()[0])  # very ugly

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        # save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'NN_w_embedding_pairwise.pkl')
        return embedding_list

    def save_trained_nets(self, name):
        """
        saves the model
        :return:
        """
        torch.save({'nn_state_dict': self.model.state_dict(),
                    'parameters': self.arguments},
                   '/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/NN_' + name + '.tar')

    def test_again(self, embedding_list):
        """
                Evaluate performance of a trained network tuned upon the alpha divergence loss.
                Note this function is called after training convergence
                :return:
                """
        num_schedules = 100
        # load in new data
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            num_schedules) + 'test_dist_early_hili_pairwise.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_data(num_schedules, data)
        # define embedding things
        embedding_list = embedding_list
        # learning rate increased
        self.opt = torch.optim.SGD(self.model.EmbeddingList.parameters(), lr=.0001)

        criterion = torch.nn.BCELoss()

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        for j in range(0, num_schedules):
            schedule_bounds = schedule_array[j]
            step = schedule_bounds[0]
            self.model.set_bayesian_embedding(embedding_list[j])
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

                        # push through nets to get preferences
                        preference_prob = self.model.forward(feature_input)
                        probability_matrix[m][n] = preference_prob[0].data.detach()[
                            0].item()  # TODO: you can do a check if only this line leads to the same thing as the line below
                        # probability_matrix[n][m] = preference_prob[0].data.detach()[1].item()

                # Set of twenty is completed
                column_vec = np.sum(probability_matrix, axis=1)

                # top 1
                highest_val = max(column_vec)
                all_indexes_that_have_highest_val = [i for i, e in enumerate(list(column_vec)) if e == highest_val]
                if len(all_indexes_that_have_highest_val) > 1:
                    print('length of indexes greater than 1: ', all_indexes_that_have_highest_val)
                # top 1
                choice = np.random.choice(all_indexes_that_have_highest_val)

                # top 3
                _, top_three = torch.topk(torch.Tensor(column_vec), 3)

                truth = Y[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1

                # index top 3
                if truth in top_three:
                    prediction_accuracy[1] += 1

                # Then do training update loop

                phi_i_num = truth + step
                phi_i = X[phi_i_num]
                phi_i_numpy = np.asarray(phi_i)
                # iterate over pairwise comparisons
                for counter in range(step, step + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = X[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_i_numpy - phi_j_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            label = Variable(torch.Tensor(torch.ones((1, 1))).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            label = Variable(torch.Tensor(torch.ones((1, 1))))

                        output = self.model(feature_input)
                        loss = criterion(output, label)
                        # prepare optimizer, compute gradient, update params

                        self.embedding_optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.embedding_optimizer.step()
                        print(self.model.EmbeddingList.state_dict())

                for counter in range(step, step + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = X[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_j_numpy - phi_i_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            label = Variable(torch.Tensor(torch.zeros((1, 1))).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            label = Variable(torch.Tensor(torch.zeros((1, 1))))

                        output = self.model.forward(feature_input)

                        self.embedding_optimizer.zero_grad()
                        loss = criterion(output, label)

                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.embedding_optimizer.step()

                # add average loss to array
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        print('top1_mean: ', np.mean(percentage_accuracy_top1))

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
            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array, set_of_twenty)

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
                    feature_input = list(feature_input)
                    feature_input.extend(self.embedding_list[which_schedule].data.numpy())
                    data_matrix.append(list(feature_input))

                    output_matrix.append(1)

            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_j_numpy - phi_i_numpy
                    feature_input = list(feature_input)
                    feature_input.extend(self.embedding_list[which_schedule].data.numpy())
                    data_matrix.append(list(feature_input))
                    output_matrix.append(0)

            if len(data_matrix) > 300000:
                return data_matrix, output_matrix


    def evaluate(self,clf, dist):

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
                        feature_input = np.concatenate([feature_input, dist[j]])
                        # push through nets
                        preference_prob = clf.predict(feature_input.reshape(1, -1))
                        probability_matrix[m][n] = preference_prob
                # feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13))

                # Set of twenty is completed
                column_vec = np.sum(probability_matrix, axis=1)
                highest_val = max(column_vec)
                all_indexes_that_have_highest_val = [i for i, e in enumerate(list(column_vec)) if e == highest_val]
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

    num_schedules = 150
    trainer = NNTrain(num_schedules)
    trainer.train()
    dists = trainer.evaluate_on_test_data()

    X, Y = trainer.generate_data()
    clf = DecisionTreeClassifier(max_depth=20)
    clf.fit(X, Y)

    y_pred = clf.predict(X)
    print(accuracy_score(Y, y_pred))

    trainer.evaluate(clf, dists)

    # X_test, Y_test = trainer.generate_test_data()
    # y_pred_test = clf.predict(X_test)
    # print(accuracy_score(Y_test, y_pred_test))

    tree.export_graphviz(clf, out_file='tree_pairwise.dot')



if __name__ == '__main__':
    main()




