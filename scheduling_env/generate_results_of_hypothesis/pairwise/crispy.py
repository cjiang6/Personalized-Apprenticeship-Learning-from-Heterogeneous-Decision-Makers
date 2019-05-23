"""
Base ProLoNet implementation
"""

import torch
import sys
import torch.nn as nn

sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')
from scheduling_env.alpha_div import AlphaLoss
from Ghost.tree_nets.vectorized_prolonet import ProLoNet
import numpy as np
from scheduling_env.argument_parser import Logger
import pickle
from torch.autograd import Variable
from utils.global_utils import save_pickle, load_in_embedding, store_embedding_back
from utils.pairwise_utils import create_new_data, create_sets_of_20_from_x_for_pairwise_comparisions, find_which_schedule_this_belongs_to
from Ghost.tree_nets.utils.deepen_prolo_supervised import deepen_with_embeddings
from Ghost.tree_nets.utils.fuzzy_to_crispy import convert_to_crisp, convert_to_complicated_crisp
sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)


# noinspection PyTypeChecker
class ProLoTrain:
    """
    class structure to train the BDT with a certain alpha.
    This class handles training the BDT, evaluating the BDT, and saving
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

        self.model = ProLoNet(input_dim=len(self.X[0]),
                              weights=None,
                              comparators=None,
                              leaves=16,
                              output_dim=2,
                              bayesian_embedding_dim=8,
                              alpha=1.5,
                              use_gpu=True,
                              vectorized=True,
                              is_value=False)

        use_gpu = True
        if use_gpu:
            self.model = self.model.cuda()
        print(self.model.state_dict())
        params = list(self.model.parameters())
        del params[0]
        self.opt = torch.optim.RMSprop([{'params': params}, {'params': self.model.bayesian_embedding, 'lr': .001}])

        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.covergence_epsilon = .01
        self.when_to_save = 1000
        self.distribution_epsilon = .0001

        self.max_depth = 10
        self.embedding_list = [torch.ones(8) * 1 / 3 for _ in range(self.num_schedules)]

    def train(self):
        """
        Trains BDT.
        Randomly samples a schedule and timestep within that schedule, produces training data using x_i - x_j
        and trains upon that.
        :return:
        """

        threshold = .1
        training_done = False
        loss_func = AlphaLoss()

        # deepening data
        deepen_data = {
            'samples': [],
            'labels': [],
            'embedding_indices': []
        }

        # variables to keep track of loss and number of tasks trained over
        running_loss_predict_tasks = 0
        num_iterations_predict_task = 0

        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(len(self.start_of_each_set_twenty))
            set_of_twenty = self.start_of_each_set_twenty[rand_timestep_within_sched]
            truth = self.Y[set_of_twenty]

            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array, set_of_twenty)
            load_in_embedding(self.model, self.embedding_list, which_schedule)

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
                    deepen_data['samples'].append(np.array(feature_input))
                    # label = add_noise_pairwise(label, self.noise_percentage)
                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]))

                    output = self.model(feature_input)
                    loss = loss_func.forward(P, output, self.alpha)

                    # NAN check (fix is in the bnn file)
                    if torch.isnan(loss):
                        print(self.alpha, ' :nan occurred at iteration ', self.total_iterations)

                    # prepare optimizer, compute gradient, update params
                    self.opt.zero_grad()
                    if loss.item() < .001 or loss.item() > 50:
                        pass
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.opt.step()

                    running_loss_predict_tasks += loss.item()
                    num_iterations_predict_task += 1

                    deepen_data['labels'].extend([0])
                    deepen_data['embedding_indices'].extend([which_schedule])

            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_j_numpy - phi_i_numpy
                    deepen_data['samples'].append(np.array(feature_input))
                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]))

                    output = self.model(feature_input)
                    loss = loss_func.forward(P, output, self.alpha)
                    # if num_iterations_predict_task % 5 == 0:
                    #     print('loss is :', loss.item())
                    # clip any very high gradients

                    # prepare optimizer, compute gradient, update params
                    self.opt.zero_grad()
                    if loss.item() < .001 or loss.item() > 50:
                        pass
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.opt.step()

                    running_loss_predict_tasks += loss.item()
                    num_iterations_predict_task += 1

                    deepen_data['labels'].extend([1])
                    deepen_data['embedding_indices'].extend([which_schedule])

            # add average loss to array
            # print(list(self.model.parameters()))

            self.embedding_list = store_embedding_back(self.model, self.embedding_list, which_schedule)
            self.total_loss_array.append(running_loss_predict_tasks / num_iterations_predict_task)
            num_iterations_predict_task = 0
            running_loss_predict_tasks = 0

            self.total_iterations += 1

            if self.total_iterations > 25 and self.total_iterations % 50 == 1:
                print('total iterations is', self.total_iterations)
                print('total loss (average for each 40, averaged)', np.mean(self.total_loss_array[-40:]))

            if self.total_iterations > 0 and self.total_iterations % self.when_to_save == self.when_to_save - 1:
                self.save_trained_nets('BDDT' + str(self.num_schedules))
                threshold -= .025

            # if self.total_iterations % 500 == 499:
            #     # self.model = deepen_with_embeddings(self.model, deepen_data, self.embedding_list, max_depth=self.max_depth, threshold=threshold)
            #     params = list(self.model.parameters())
            #     del params[0]
            #     self.opt = torch.optim.RMSprop([{'params': params}, {'params': self.model.bayesian_embedding, 'lr': .001}])
                # deepen_data = {
                #     'samples': [],
                #     'labels': [],
                #     'embedding_indices': []
                # }

            if self.total_iterations > 2500 and np.mean(self.total_loss_array[-100:]) - np.mean(
                    self.total_loss_array[-500:]) < self.covergence_epsilon:
                training_done = True

                self.model = convert_to_crisp(self.model, None)

    def evaluate_on_test_data(self, load_in_model=False):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.

        Note this function is called after training convergence
        :return:
        """
        # define new optimizer that only optimizes gradient
        num_schedules = 75
        loss_func = AlphaLoss()
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/test/' + str(
            num_schedules) + '_inf_hetero_deadline_pairwise.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_data(num_schedules, data)
        start_of_each_set_twenty = create_sets_of_20_from_x_for_pairwise_comparisions(X)

        embedding_optimizer = torch.optim.RMSprop([{'params': self.model.bayesian_embedding, 'lr': .001}])
        embedding_list = [torch.ones(8) * 1 / 3 for _ in range(num_schedules)]

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        if load_in_model:
            self.model.load_state_dict(torch.load('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/model_homog.tar')['nn_state_dict'])

        for j in range(0, num_schedules):
            schedule_bounds = schedule_array[j]
            step = schedule_bounds[0]
            load_in_embedding(self.model, embedding_list, j)
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
                        preference_prob = self.model.forward(feature_input)
                        probability_matrix[m][n] = preference_prob[0].data.detach()[0].item()
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

                        output = self.model(feature_input)
                        loss = loss_func.forward(P, output, self.alpha)
                        # prepare optimizer, compute gradient, update params
                        if loss.item() < .001 or loss.item() > 50:
                            pass
                        else:
                            embedding_optimizer.zero_grad()
                            if loss.item() < .001 or loss.item() > 50:
                                pass
                            else:
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

                        output = self.model(feature_input)
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
                embedding_list = store_embedding_back(self.model, embedding_list, j)
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'crispy_inf_BDT' + str(self.num_schedules))

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
    num_schedules = 150
    trainer = ProLoTrain(num_schedules)
    trainer.train()
    trainer.evaluate_on_test_data()


if __name__ == '__main__':
    main()
