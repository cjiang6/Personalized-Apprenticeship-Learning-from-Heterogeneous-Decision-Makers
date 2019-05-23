"""
Base ProLoNet implementation
"""

import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')
from scheduling_env.alpha_div import AlphaLoss
from base_testing_environment.prolonet import ProLoNet

import numpy as np
from scheduling_env.argument_parser import Logger
import pickle
from torch.autograd import Variable
from utils.global_utils import save_pickle, load_in_embedding, store_embedding_back
from utils.naive_utils import create_new_dataset, find_which_schedule_this_belongs_to

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
            self.num_schedules) + 'dist_early_hili_naive.pkl'

        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_dataset(num_schedules=self.num_schedules, data=self.data)
        for i, each_element in enumerate(self.X):
            self.X[i] = each_element + list(range(20))

        self.model = ProLoNet(input_dim=len(self.X[0]),
                              weights=None,
                              comparators=None,
                              leaves=64,
                              output_dim=20,
                              bayesian_embedding_dim=8,
                              alpha=1.5,
                              use_gpu=True,
                              vectorized=True,
                              is_value=False)

        use_gpu = True
        if use_gpu:
            self.model = self.model.cuda()
        print(self.model.state_dict())
        self.opt = torch.optim.RMSprop([{'params': list(self.model.parameters())[:-1]}, {'params': self.model.bayesian_embedding.parameters(), 'lr': .01}])

        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.covergence_epsilon = .01
        self.when_to_save = 1000
        self.distribution_epsilon = .0001

        self.max_depth = 10
        self.embedding_list = [torch.ones(8) * 1 / 3 for _ in range(self.num_schedules)]

    def train(self):
        """
        Trains PDDT.
        :return:
        """

        threshold = .05
        training_done = False

        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(len(self.X))
            input_nn = self.X[rand_timestep_within_sched]
            truth_nn = self.Y[rand_timestep_within_sched]

            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array, rand_timestep_within_sched)
            self.model.set_bayesian_embedding(self.embedding_list[which_schedule])

            if torch.cuda.is_available():
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())  # change to 5 to increase batch size
                truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
            else:
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))
                truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())

            self.opt.zero_grad()
            output = self.model.forward(input_nn)
            loss = F.cross_entropy(output, truth)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.opt.step()

            self.embedding_list[which_schedule] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

            # add average loss to array
            # print(list(self.model.parameters()))

            self.total_loss_array.append(loss.item())
            self.total_iterations += 1

            if self.total_iterations > 25 and self.total_iterations % 50 == 1:
                print('total iterations is', self.total_iterations)
                print('total loss (average for each 40, averaged)', np.mean(self.total_loss_array[-40:]))
                # print(self.model.state_dict())

            if self.total_iterations > 0 and self.total_iterations % self.when_to_save == self.when_to_save - 1:
                self.save_trained_nets('PDDT' + str(self.num_schedules))
                threshold -= .1

            if self.total_iterations > 100000 and np.mean(self.total_loss_array[-100:]) - np.mean(
                    self.total_loss_array[-500:]) < self.covergence_epsilon:
                training_done = True

    def evaluate_on_test_data(self, load_in_model=False):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.

        Note this function is called after training convergence
        :return:
        """
        # define new optimizer that only optimizes gradient
        num_schedules = 100
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            num_schedules) + 'test_dist_early_hili_naive.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_dataset(num_schedules=num_schedules, data=data)
        for i, each_element in enumerate(X):
            X[i] = each_element + list(range(20))

        embedding_optimizer = torch.optim.SGD([{'params': self.model.bayesian_embedding.parameters()}], lr=.1)
        embedding_list = [torch.ones(8) * 1 / 3 for _ in range(num_schedules)]

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        if load_in_model:
            self.model.load_state_dict(torch.load('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/model_homog.tar')['nn_state_dict'])

        for i, schedule in enumerate(schedule_array):
            self.model.set_bayesian_embedding(self.embedding_list[i])

            for count in range(schedule[0], schedule[1] + 1):

                net_input = X[count]
                truth = Y[count]

                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)).cuda())  # change to 5 to increase batch size
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)))
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).long())

                #####forward#####
                output = self.model.forward(input_nn)
                embedding_optimizer.zero_grad()
                loss = F.cross_entropy(output, truth)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                embedding_optimizer.step()

                index = torch.argmax(output).item()

                # top 3
                _, top_three = torch.topk(output, 3)

                if index == truth.item():
                    prediction_accuracy[0] += 1

                if truth.item() in top_three.detach().cpu().tolist()[0]:
                    prediction_accuracy[1] += 1

            # add average loss to array
            embedding_list[i] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', i)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'DDT_w_embedding')

    def save_trained_nets(self, name):
        """
        saves the model
        :return:
        """
        torch.save({'nn_state_dict': self.model.state_dict(),
                    'parameters': self.arguments},
                   '/home/ghost/PycharmProjects/bayesian_prolo/saved_models/naive_saved_models/BNN_' + name + '.tar')

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
        save_pickle(file=data, file_location=self.home_dir + '/saved_models/naive_saved_models/', special_string=special_string)


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
