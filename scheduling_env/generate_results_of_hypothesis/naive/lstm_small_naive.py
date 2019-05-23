"""
training and evaluating an lstm with the same number of parameters as a BDDT
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
from utils.naive_utils import create_new_dataset, find_which_schedule_this_belongs_to

sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)


class LSTMSmall(nn.Module):
    def __init__(self):
        super(LSTMSmall, self).__init__()
        self.fc1 = nn.Linear(242, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(64, 64)
        self.relu21 = nn.ReLU()
        self.fc3 = nn.Linear(64, 20)
        self.soft3 = nn.Softmax()
        self.LSTM_layer = nn.LSTMCell(input_size=64, hidden_size=64)
        self.hidden = (torch.randn(1, 64), torch.randn(1, 64))

    def forward(self, x, previous_hidden_state):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc21(x)
        x = self.relu21(x)
        x, c_x = self.LSTM_layer(x, previous_hidden_state)
        self.hidden = (x, c_x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.soft3(x)
        return x

    def reinitialize_hidden_to_random(self):
        self.hidden = (torch.randn(1, 64), torch.randn(1, 64))


# noinspection PyTypeChecker
class LSTMTrain:
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
            self.num_schedules) + '_inf_hetero_deadline_naive.pkl'

        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_dataset(num_schedules=self.num_schedules, data=self.data)
        for i, each_element in enumerate(self.X):
            self.X[i] = each_element + list(range(20))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LSTMSmall().to(device)

        print(self.model.state_dict())
        params = list(self.model.parameters())
        self.opt = torch.optim.Adam(params, lr=.0001)
        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.convergence_epsilon = .01
        self.when_to_save = 1000
        self.distribution_epsilon = .0001

    def train(self):
        """
        Trains LSTM.
        Randomly samples a schedule and timestep within that schedule, produces training data using x_i - x_j
        and trains upon that.
        :return:
        """
        timesteps = None
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
                rand_timestep_within_sched = np.random.randint(len(self.X))
                which_schedule = find_which_schedule_this_belongs_to(self.schedule_array, rand_timestep_within_sched)
                if rand_timestep_within_sched + 2 > self.schedule_array[which_schedule][1]:
                    pass
                else:
                    found_a_suitable_candidate = True
                    timesteps = [rand_timestep_within_sched, rand_timestep_within_sched + 1, rand_timestep_within_sched + 2]
                    self.model.reinitialize_hidden_to_random()

            for timestep in timesteps:
                truth = self.Y[timestep]
                previous_hidden_state = tuple([t.detach().cuda() for t in self.model.hidden])

                input_nn = self.X[timestep]
                truth_nn = self.Y[timestep]

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
                output = self.model.forward(input_nn, previous_hidden_state)

                loss = loss_func.forward(P, output, self.alpha)
                if loss.item() < .05 or loss.item() > 5:
                    pass
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.opt.step()
                self.total_loss_array.append(loss.item())

            self.total_iterations += 1

            if self.total_iterations > 25 and self.total_iterations % 50 == 1:
                print('total iterations is', self.total_iterations)
                print('total loss (average for each 40, averaged)', np.mean(self.total_loss_array[-40:]))

            if self.total_iterations > 0 and self.total_iterations % self.when_to_save == self.when_to_save - 1:
                self.save_trained_nets('lstm_small' + str(self.num_schedules))

            if self.total_iterations > 11000 and np.mean(self.total_loss_array[-100:]) - np.mean(
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
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/test/' + str(
            num_schedules) + '_inf_hetero_deadline_naive.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_dataset(num_schedules=num_schedules, data=data)
        for i, each_element in enumerate(X):
            X[i] = each_element + list(range(20))

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        if load_in_model:  # TODO: somehow get the string when the update_model flag is true
            self.model.load_state_dict(torch.load('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/NN_homog.tar')['nn_state_dict'])

        for i, schedule in enumerate(schedule_array):
            self.model.reinitialize_hidden_to_random()

            for count in range(schedule[0], schedule[1] + 1):
                previous_hidden_state = tuple([t.detach().cuda() for t in self.model.hidden])
                net_input = X[count]
                truth = Y[count]

                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).cuda().long())
                    P = Variable(torch.Tensor(np.ones((1, 20)))).cuda()
                    P *= self.distribution_epsilon
                    P[0][truth] = 1 - 19 * self.distribution_epsilon
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)))
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)))
                    P = Variable(torch.Tensor(np.ones((1, 20))))
                    P *= self.distribution_epsilon
                    P[0][truth] = 1 - 19 * self.distribution_epsilon

                #####forward#####
                output = self.model.forward(input_nn, previous_hidden_state)

                index = torch.argmax(output).item()

                # top 3
                _, top_three = torch.topk(output, 3)

                if index == truth.item():
                    prediction_accuracy[0] += 1

                if truth.item() in top_three.detach().cpu().tolist()[0]:
                    prediction_accuracy[1] += 1


            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', i)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'inf_lstm_small_' + str(self.num_schedules))

    def save_trained_nets(self, name):
        """
        saves the model
        :return:
        """
        torch.save({'nn_state_dict': self.model.state_dict(),
                    'parameters': self.arguments},
                   '/home/ghost/PycharmProjects/bayesian_prolo/saved_models/naive_saved_models/BLSTM_' + name + '.tar')

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
    for num_schedules in (3, 9, 15, 150, 1500):
        trainer = LSTMTrain(num_schedules)
        trainer.train()
        trainer.evaluate_on_test_data()


if __name__ == '__main__':
    main()
