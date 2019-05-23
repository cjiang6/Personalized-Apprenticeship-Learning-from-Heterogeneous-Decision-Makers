"""
Sweep on naive data to find the optimal embedding size
"""

import torch
import sys

# sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')
from scheduling_env.alpha_div import AlphaLoss
from Ghost.tree_nets.vectorized_prolonet import ProLoNet
import numpy as np
from scheduling_env.argument_parser import Logger
import pickle
from torch.autograd import Variable

from utils.naive_utils import create_new_dataset

sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)
from utils.global_utils import save_pickle, load_in_embedding, store_embedding_back


# noinspection PyTypeChecker
class BDTTrain:
    """
    class structure to train the BDT with a certain alpha.
    This class handles training the BDT, evaluating the BDT, and saving
    """

    def __init__(self, bayesian_dim):
        self.arguments = Logger()
        self.alpha = .9
        self.num_schedules = 150  
        self.home_dir = self.arguments.home_dir
        self.total_loss_array = []

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + '_hetero_deadline_naive.pkl'

        self.bayesian_embedding_dim = int(bayesian_dim)
        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_dataset(self.data, num_schedules=self.num_schedules)
        for i, each_element in enumerate(self.X):
            self.X[i] = each_element + list(range(20))

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        use_gpu = True
        self.model = ProLoNet(input_dim=len(self.X[0]),
                              weights=None,
                              comparators=None,
                              leaves=4,
                              output_dim=20,
                              bayesian_embedding_dim=self.bayesian_embedding_dim,
                              alpha=1.5,
                              use_gpu=use_gpu,
                              vectorized=False,
                              is_value=False)

        self.model_NLL = ProLoNet(input_dim=len(self.X[0]),
                                  weights=None,
                                  comparators=None,
                                  leaves=4,
                                  output_dim=20,
                                  bayesian_embedding_dim=self.bayesian_embedding_dim,
                                  alpha=1.5,
                                  use_gpu=use_gpu,
                                  vectorized=False,
                                  is_value=False)

        if use_gpu:
            self.model = self.model.cuda()
            self.model_NLL = self.model_NLL.cuda()

        print(self.model.state_dict())
        params = list(self.model.parameters())
        del params[0]
        self.opt = torch.optim.RMSprop([{'params': params}, {'params': self.model.bayesian_embedding, 'lr': .001}])

        params = list(self.model_NLL.parameters())
        del params[0]
        self.opt2 = torch.optim.RMSprop([{'params': params}, {'params': self.model_NLL.bayesian_embedding, 'lr': .001}])

        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.covergence_epsilon = .01
        self.when_to_save = 1000
        self.distribution_epsilon = .0001

        self.embedding_list = [torch.ones(self.bayesian_embedding_dim) * 1 / 3 for _ in range(self.num_schedules)]
        self.embedding_list_NLL = [torch.ones(self.bayesian_embedding_dim) * 1 / 3 for _ in range(self.num_schedules)]

    def train(self):
        """
        Trains BDT.
        Randomly samples a schedule and timestep within that schedule, and passes in the corresponding data in an attempt to classify which task was scheduled
        :return:
        """
        criterion = torch.nn.CrossEntropyLoss()
        loss_func = AlphaLoss()
        training_done = False
        cv_cutoff = .8 * len(self.X)

        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(cv_cutoff)
            input_nn = self.X[rand_timestep_within_sched]
            truth_nn = self.Y[rand_timestep_within_sched]

            which_schedule = self.find_which_schedule_this_belongs_to(rand_timestep_within_sched)
            load_in_embedding(self.model, self.embedding_list, which_schedule)
            load_in_embedding(self.model_NLL, self.embedding_list_NLL, which_schedule)

            # iterate over pairwise comparisons
            if torch.cuda.is_available():
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())  # change to 5 to increase batch size
                P = Variable(torch.Tensor(np.ones((1, 20)))).cuda()
                P *= self.distribution_epsilon
                P[0][truth_nn] = 1 - 19 * self.distribution_epsilon
                truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
            else:
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))
                P = Variable(torch.Tensor(np.ones((1, 20) * self.distribution_epsilon)))
                P[0][truth_nn] = 1 - 19 * self.distribution_epsilon
                truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())

            self.opt.zero_grad()
            output = self.model.forward(input_nn)
            loss = loss_func.forward(P, output, self.alpha)
            loss.backward()
            self.opt.step()

            self.opt2.zero_grad()
            output_nn = self.model_NLL.forward(input_nn)
            loss_nn = criterion(output_nn, truth)
            loss_nn.backward()
            self.opt2.step()

            self.total_loss_array.append(loss.item())
            store_embedding_back(self.model, self.embedding_list, which_schedule)
            store_embedding_back(self.model_NLL, self.embedding_list_NLL, which_schedule)
            total_iterations = len(self.total_loss_array)

            if total_iterations % 50 == 49:
                print('loss at', total_iterations, 'is', loss.item())
                print('loss_NN at', total_iterations, 'is', loss_nn.item())

            if total_iterations > 100000:
                training_done = True

    def evaluate_alpha(self):

        """
        Evaluate performance of a trained network.
        This is tested on 20% of the data and will be stored in a text file.
        :return:
        """

        opt = torch.optim.RMSprop([{'params': self.model.bayesian_embedding, 'lr': .001}])
        loss_func = AlphaLoss()

        percentage_accuracy_top1 = []

        for i, schedule in enumerate(self.schedule_array):
            if i < .8 * len(self.schedule_array):
                continue
            load_in_embedding(self.model, self.embedding_list, i)
            prediction_accuracy = 0
            for count in range(schedule[0], schedule[1] + 1):

                input_nn = self.X[count]
                truth_nn = self.Y[count]

                # if torch.cuda.is_available():
                #     input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)).cuda())
                #     truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).cuda().long())
                # else:
                #     input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)))
                #     truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)))

                # iterate over pairwise comparisons
                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())  # change to 5 to increase batch size
                    P = Variable(torch.Tensor(np.ones((1, 20)))).cuda()
                    P *= self.distribution_epsilon
                    P[0][truth_nn] = 1 - 19 * self.distribution_epsilon
                    truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))
                    P = Variable(torch.Tensor(np.ones((1, 20) * self.distribution_epsilon)))
                    P[0][truth_nn] = 1 - 19 * self.distribution_epsilon
                    truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())

                opt.zero_grad()
                output = self.model.forward(input_nn)
                loss = loss_func.forward(P, output, self.alpha)
                loss.backward()
                opt.step()

                index = torch.argmax(output).item()

                if index == truth.item():
                    prediction_accuracy += 1

            print('Prediction Accuracy: top1: ', prediction_accuracy / 20)
            store_embedding_back(self.model, self.embedding_list, i)
            print('schedule num:', i)
            percentage_accuracy_top1.append(prediction_accuracy / 20)
            self.save_performance_results(percentage_accuracy_top1)
        print(np.mean(percentage_accuracy_top1))

    def evaluate_other(self):
        """
        Evaluate performance of a trained network.
        This is tested on 20% of the data and will be stored in a text file.
        :return:
        """

        opt = torch.optim.RMSprop([{'params': self.model_NLL.bayesian_embedding, 'lr': .001}])
        criterion = torch.nn.CrossEntropyLoss()

        percentage_accuracy_top1 = []

        for i, schedule in enumerate(self.schedule_array):
            if i < .8 * len(self.schedule_array):
                continue
            load_in_embedding(self.model, self.embedding_list, i)
            prediction_accuracy = 0
            for count in range(schedule[0], schedule[1] + 1):

                input_nn = self.X[count]
                truth_nn = self.Y[count]

                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))
                    truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)))

                opt.zero_grad()
                output_nn = self.model_NLL.forward(input_nn)
                loss_nn = criterion(output_nn, truth)
                loss_nn.backward()
                opt.step()

                index = torch.argmax(output_nn).item()

                if index == truth.item():
                    prediction_accuracy += 1

            print('Prediction Accuracy: top1: ', prediction_accuracy / 20)
            store_embedding_back(self.model_NLL, self.embedding_list, i)
            print('schedule num:', i)
            percentage_accuracy_top1.append(prediction_accuracy / 20)
            self.save_performance_results(percentage_accuracy_top1)
        print(np.mean(percentage_accuracy_top1))

    def save_performance_results(self, top1):
        """
        saves performance of top1 and top3
        :return:
        """
        print('top1_mean for ', self.alpha, ' is : ', np.mean(top1))
        data = {'top1_mean': np.mean(top1),
                'top1_stderr': np.std(top1) / np.sqrt(len(top1)),
                'embedding': self.bayesian_embedding_dim}
        save_pickle(file=data, file_location=self.home_dir + '/saved_models/naive_saved_models/', special_string=str(self.bayesian_embedding_dim) + 'baydim_naivetest.pkl')

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


def main():
    """
    entry point for file
    :return:
    """
    trainer = None
    for bayesian_embedding_dim in np.linspace(1, 16, num=16):
        trainer = BDTTrain(8)
        trainer.train()
        trainer.evaluate_alpha()
        trainer.evaluate_other()
        if bayesian_embedding_dim == 6:
            exit()

    # trainer.find_optimal_embedding()


if __name__ == '__main__':
    main()
