"""
Testing the blstm_small
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
from utils.pairwise_utils import create_new_data, create_sets_of_20_from_x_for_pairwise_comparisions, find_which_schedule_this_belongs_to
from utils.global_utils import load_in_embedding_bnn, store_embedding_back_bnn

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
        self.embedding = nn.Parameter(torch.randn(1, 8))

    def forward(self):
        """
        doesn't do anything
        :return:
        """
        return


# noinspection PyTypeChecker
class BLSTMSmall(nn.Module):
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
        super(BLSTMSmall, self).__init__()
        self.fc1 = nn.Linear(13, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(24, 16)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(16, 16)
        self.relu21 = nn.ReLU()
        self.fc3 = nn.Linear(16, 2)
        self.soft3 = nn.Softmax()
        self.LSTM_layer = nn.LSTMCell(input_size=16, hidden_size=16)
        self.hidden = (torch.randn(1, 16), torch.randn(1, 16))
        self.EmbeddingList = nn.ModuleList(EmbeddingModule() for _ in range(1))

    def forward(self, x, previous_hidden_state):
        """
        forward pass through the network
        :param x: state
        :param previous_hidden_state: time variable
        :return:
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x, c_x = self.LSTM_layer(x, previous_hidden_state)
        self.hidden = (x, c_x)
        w = self.EmbeddingList[0].embedding
        x = torch.cat([x, w], dim=1)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc21(x)
        x = self.relu21(x)
        x = self.fc3(x)
        x = self.soft3(x)

        return x

    def reinitialize_hidden_to_random(self):
        """
        initialize to random for every new schedule
        :return:
        """
        self.hidden = (torch.randn(1, 16), torch.randn(1, 16))


# noinspection PyTypeChecker
class BLSTMTrain:
    """
    class structure to train the NN for a certain amount of schedules.
    This class handles training the NN, evaluating the NN, and saving the results
    """

    def __init__(self, num_schedules):
        self.arguments = Logger()
        self.alpha = .9
        self.num_schedules = num_schedules
        self.home_dir = self.arguments.home_dir
        self.total_loss_array = []

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + '_inf_hetero_deadline_pairwise.pkl'

        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_data(self.num_schedules, self.data)
        self.start_of_each_set_twenty = create_sets_of_20_from_x_for_pairwise_comparisions(self.X)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = BLSTMSmall().to(device)

        print(self.model.state_dict())
        bnn_params = list(self.model.parameters())
        del bnn_params[10]
        self.opt = torch.optim.Adam([{'params': bnn_params}])  # TODO: try together and not together
        self.embedding_optimizer = torch.optim.SGD(self.model.EmbeddingList.parameters(), lr=.001)
        # params = list(self.model.parameters())
        # self.opt = torch.optim.Adam(params)
        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.convergence_epsilon = .01
        self.when_to_save = 1000
        self.distribution_epsilon = .0001

        self.embedding_list = [torch.ones(1, 8) * 1/3 for _ in range(self.num_schedules)]

    def train(self):
        """
        Trains NN.
        Randomly samples a schedule and timestep within that schedule, produces training data using x_i - x_j
        and trains upon that.
        :return:
        """
        sets_of_twenty = None
        which_schedule = None
        training_done = False
        loss_func = AlphaLoss()

        # variables to keep track of loss and number of tasks trained over
        running_loss_predict_tasks = 0
        num_iterations_predict_task = 0
        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            # Quick Fix
            found_a_suitable_candidate = False
            while not found_a_suitable_candidate:
                rand_timestep_within_sched = np.random.randint(len(self.start_of_each_set_twenty))
                set_of_twenty = self.start_of_each_set_twenty[rand_timestep_within_sched]
                which_schedule = find_which_schedule_this_belongs_to(self.schedule_array, set_of_twenty)
                if set_of_twenty + 59 > self.schedule_array[which_schedule][1]:
                    pass
                else:
                    found_a_suitable_candidate = True
                    sets_of_twenty = [set_of_twenty, set_of_twenty + 20, set_of_twenty + 40]
                    self.model.reinitialize_hidden_to_random()
                    load_in_embedding_bnn(self.model, self.embedding_list, which_schedule)

            # 3 schedules
            for set_of_twenty in sets_of_twenty:
                truth = self.Y[set_of_twenty]

                # find feature vector of true action taken
                phi_i_num = truth + set_of_twenty
                phi_i = self.X[phi_i_num]
                phi_i_numpy = np.asarray(phi_i)
                # changes each timestep
                previous_hidden_state = tuple([t.detach().cuda() for t in self.model.hidden])
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
                            P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]))

                        output = self.model.forward(feature_input, previous_hidden_state)
                        self.opt.zero_grad()
                        self.embedding_optimizer.zero_grad()
                        loss = loss_func.forward(P, output, self.alpha)
                        if torch.isnan(loss):
                            print(self.alpha, ' :nan occurred at iteration ', self.total_iterations)

                        if loss.item() < .001 or loss.item() > 55:
                            pass
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                            self.opt.step()
                            self.embedding_optimizer.step()

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
                            P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]))

                        output = self.model.forward(feature_input, previous_hidden_state)
                        self.opt.zero_grad()
                        self.embedding_optimizer.zero_grad()
                        loss = loss_func.forward(P, output, self.alpha)
                        if torch.isnan(loss):
                            print(self.alpha, ' :nan occurred at iteration ', self.total_iterations, ' at', num_iterations_predict_task)

                        if loss.item() < .001 or loss.item() > 55:
                            pass
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                            self.opt.step()
                            self.embedding_optimizer.step()

                        running_loss_predict_tasks += loss.item()

                        num_iterations_predict_task += 1

                self.embedding_list = store_embedding_back_bnn(self.model, self.embedding_list, which_schedule)
                self.total_loss_array.append(running_loss_predict_tasks / num_iterations_predict_task)
                num_iterations_predict_task = 0
                running_loss_predict_tasks = 0

            self.total_iterations += 1

            if self.total_iterations > 25 and self.total_iterations % 50 == 1:
                print('total iterations is', self.total_iterations)
                print('total loss (average for each 40, averaged)', np.mean(self.total_loss_array[-40:]))

            if self.total_iterations > 0 and self.total_iterations % self.when_to_save == self.when_to_save - 1:
                self.save_trained_nets('bnn_small' + str(self.num_schedules))

            if self.total_iterations > 2000 and np.mean(self.total_loss_array[-100:]) - np.mean(
                    self.total_loss_array[-500:]) < self.convergence_epsilon:
                training_done = True

    def evaluate_on_test_data(self, load_in_model=False):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        Note this function is called after training convergence
        :return:
        """
        num_schedules = 75
        # load in new data
        loss_func = AlphaLoss()
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/test/' + str(
            num_schedules) + '_inf_hetero_deadline_pairwise.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_data(num_schedules, data)
        start_of_each_set_twenty = create_sets_of_20_from_x_for_pairwise_comparisions(X)

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        embedding_optimizer = torch.optim.Adam(self.model.EmbeddingList.parameters(), lr=.001)

        embedding_list = [torch.ones(1, 8) * 1/3 for i in range(num_schedules)]

        if load_in_model:  # TODO: somehow get the string when the update_model flag is true
            self.model.load_state_dict(torch.load('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/NN_homog.tar')['nn_state_dict'])

        for j in range(0, num_schedules):
            schedule_bounds = schedule_array[j]
            step = schedule_bounds[0]
            load_in_embedding_bnn(self.model, embedding_list, j)
            self.model.reinitialize_hidden_to_random()
            while step < schedule_bounds[1]:
                probability_matrix = np.zeros((20, 20))
                previous_hidden_state = tuple([t.detach().cuda() for t in self.model.hidden])

                for m, counter in enumerate(range(step, step + 20)):
                    phi_i = X[counter]
                    phi_i_numpy = np.asarray(phi_i)

                    # for each set of twenty
                    for n, second_counter in enumerate(range(step, step + 20)):
                        # fill entire array with diagonals set to zero
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
                        preference_prob = self.model.forward(feature_input, previous_hidden_state)
                        probability_matrix[m][n] = preference_prob[0].data.detach()[0].item() # TODO: you can do a check if only this line leads to the same thing as the line below
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

                # forward
                phi_i_num = truth + step  # old method: set_of_twenty[0] + truth
                phi_i = X[phi_i_num]
                phi_i_numpy = np.asarray(phi_i)
                # iterate over pairwise comparisons
                for counter in range(step, step + 20):
                    if counter == phi_i_num:  # if counter == phi_i_num:
                        continue
                    else:
                        phi_j = X[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_i_numpy - phi_j_numpy

                        # label = add_noise_pairwise(label, self.noise_percentage)
                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]))

                        output = self.model(feature_input, previous_hidden_state)
                        loss = loss_func.forward(P, output, self.alpha)
                        # prepare optimizer, compute gradient, update params
                        if loss.item() < .001 or loss.item() > 50:
                            pass
                        else:
                            embedding_optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                            embedding_optimizer.step()

                for counter in range(step, step + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = X[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_j_numpy - phi_i_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                            P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                            P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]))

                        output = self.model(feature_input, previous_hidden_state)
                        loss = loss_func.forward(P, output, self.alpha)
                        # print('loss is :', loss.item())
                        # clip any very high gradients

                        # prepare optimizer, compute gradient, update params
                        if loss.item() < .001 or loss.item() > 50:
                            pass
                        else:
                            embedding_optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                            embedding_optimizer.step()

                # add average loss to array
                store_embedding_back_bnn(self.model, embedding_list, j)
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'inf_blstm_small_' + str(self.num_schedules))

    def save_trained_nets(self, name):
        """
        saves the model
        :return:
        """
        torch.save({'nn_state_dict': self.model.state_dict(),
                    'parameters': self.arguments},
                   '/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/BNN_' + name + '.tar')

    def save_performance_results(self, top1, top3, special_string):
        """
        saves performance of top1 and top3
        :return:
        """
        print('top1_mean for ', self.alpha, ' is : ', np.mean(top1))
        data = {'top1_mean': np.mean(top1),
                'top3_mean': np.mean(top3),
                'top1_stderr': np.std(top1) / np.sqrt(len(top1)),
                'top3_stderr': np.std(top3) / np.sqrt(len(top3))}
        save_pickle(file=data, file_location=self.home_dir + '/saved_models/pairwise_saved_models/', special_string=special_string)


def main():
    """
    entry point for file
    :return:
    """
    for num_schedules in (3, 9, 15, 150, 1500):
        trainer = BLSTMTrain(num_schedules)
        trainer.train()
        trainer.evaluate_on_test_data()


if __name__ == '__main__':
    main()
