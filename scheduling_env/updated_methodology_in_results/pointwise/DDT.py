"""
File to check performance on a heterogeneous dataset. It should be near 100
"""

import torch
import sys
import torch.nn as nn

sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')
from base_testing_environment.prolonet import ProLoNet
import numpy as np
from scheduling_env.argument_parser import Logger
import pickle
from torch.autograd import Variable
from utils.global_utils import save_pickle
from utils.pairwise_utils import create_new_data, create_sets_of_20_from_x_for_pairwise_comparisions

sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)


class ProLoTrain:
    """
    class structure to train the BDT with a certain alpha.
    This class handles training the BDT, evaluating the BDT, and saving
    """

    def __init__(self,num_schedules):
        self.arguments = Logger()
        self.alpha = .9
        self.num_schedules = num_schedules
        self.home_dir = self.arguments.home_dir
        self.total_loss_array = []

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + 'dist_early_hili_pairwise.pkl'

        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_data(self.num_schedules, self.data)
        self.start_of_each_set_twenty = create_sets_of_20_from_x_for_pairwise_comparisions(self.X)

        self.model = ProLoNet(input_dim=len(self.X[0]),
                              weights=None,
                              comparators=None,
                              leaves=16,
                              output_dim=1,
                              bayesian_embedding_dim=None,
                              alpha=1.5,
                              use_gpu=True,
                              vectorized=True,
                              is_value=True)

        use_gpu = True
        if use_gpu:
            self.model = self.model.cuda()
        print(self.model.state_dict())
        self.opt = torch.optim.RMSprop(self.model.parameters())

        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.covergence_epsilon = .01
        self.when_to_save = 1000
        self.distribution_epsilon = .0001


    def train(self):
        """
        Trains BDT.
        Randomly samples a schedule and timestep within that schedule, produces training data using x_i - x_j
        and trains upon that.
        :return:
        """
        # loss = nn.CrossEntropyLoss()

        training_done = False
        criterion = torch.nn.BCELoss()

        # variables to keep track of loss and number of tasks trained over
        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            set_of_twenty = np.random.choice(self.start_of_each_set_twenty)
            truth = self.Y[set_of_twenty]

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

                output = self.model.forward(feature_input)
                sig = torch.nn.Sigmoid()
                output = sig(output)

                self.opt.zero_grad()
                loss = criterion(output, label)
                if counter == phi_i_num:
                    loss *= 25
                # print(self.total_iterations)
                if torch.isnan(loss):
                    print('nan occurred at iteration ', self.total_iterations, ' at', num_iterations_predict_task)
                loss.backward()
                # print(self.model.state_dict())
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.opt.step()
                running_loss_predict_tasks += loss.item()
                num_iterations_predict_task += 1

            # add average loss to array
            # print(list(self.model.parameters()))

            self.total_loss_array.append(running_loss_predict_tasks / num_iterations_predict_task)


            self.total_iterations += 1

            if self.total_iterations % 500 == 499:
                print('total loss (average for each 40, averaged) at iteration ', self.total_iterations, ' is ', np.mean(self.total_loss_array[-40:]))

            if self.total_iterations > 10000 and np.mean(self.total_loss_array[-100:]) - np.mean(
                    self.total_loss_array[-500:]) < self.covergence_epsilon:
                training_done = True


    def evaluate_on_test_data(self, model, load_in_model=False):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        This is tested on 20% of the data and will be stored in a text file.
        Note this function is called after training convergence
        :return:
        """
        # define new optimizer that only optimizes gradient
        num_schedules = 100
        # load in new data
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            num_schedules) + 'test_dist_early_hili_pairwise.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_data(num_schedules, data)

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        if load_in_model:
            model.load_state_dict(torch.load('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/model_homog.tar')['nn_state_dict'])


        for j in range(0, num_schedules):
            schedule_bounds = schedule_array[j]
            step = schedule_bounds[0]
            while step < schedule_bounds[1]:
                probability_vector = np.zeros((1, 20))

                for m, counter in enumerate(range(step, step + 20)):
                    phi_i = X[counter]
                    phi_i_numpy = np.asarray(phi_i)



                    feature_input = phi_i_numpy

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))

                    # push through nets
                    preference_prob = model.forward(feature_input)
                    sig = torch.nn.Sigmoid()
                    preference_prob = sig(preference_prob)
                    probability_vector[0][m] = preference_prob[0].data.detach()[
                        0].item()
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

                # add average loss to array
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'DDT_pointwise'+ str(self.num_schedules))


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
    trainer.evaluate_on_test_data(trainer.model)



if __name__ == '__main__':
    main()
