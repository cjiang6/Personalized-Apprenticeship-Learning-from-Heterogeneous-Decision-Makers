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
from utils.global_utils import save_pickle
from utils.naive_utils import create_new_dataset

sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)


class NNSmall(nn.Module):
    """
    number of parameters should be it is N*(|X| + |E| + |C|) + |Y|*L
    where N=num_nodes, X=input_sample, E=bayesian_embedding, Y=output_classes, L=num_leaves, C=comparator_vector (1 for non_vec, |X| for vec)
    I will consider a baseline case of 64 nodes, 8 is the size of bayesian embedding, 13 is the size of the input, L is 6, Y is 2. Comparators is 13.
    # updated: num_nodes*(3*input_size + 3*embedding_size) + num_leaves*output_size,
    so N*(|X| + |E| + |C| + |S|) + Y*L where N=num_nodes, X=input_sample, E=bayesian_embedding, C=comparator_vector, S=selector_vector, Y=output_classes, L=num_leaves
    but |C| = |S| = (|X|+|E|)
    so it's N*(3*(|X|+|E|))

    In total this is 67086
    This one has 3649 (little smaller but I think its close enough) # TODO: maybe add one more layer
    NOTE: this line returns number of model params  pytorch_total_params = sum(p.numel() for p in self.model.parameters()),
    NOTE: only trainable params is pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    """

    def __init__(self):
        super(NNSmall, self).__init__()
        self.fc1 = nn.Linear(242, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(128, 64)
        self.relu21 = nn.ReLU()
        self.fc22 = nn.Linear(64, 32)
        self.relu22 = nn.ReLU()
        self.fc23 = nn.Linear(32, 32)
        self.relu23 = nn.ReLU()
        self.fc3 = nn.Linear(32, 20)
        self.soft = nn.Softmax()

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
        x = self.fc23(x)
        x = self.relu23(x)
        x = self.fc3(x)
        x = self.soft(x)

        return x


# noinspection PyTypeChecker
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
        self.total_loss_array = []

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + '_inf_hetero_deadline_naive.pkl'

        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_dataset(num_schedules=self.num_schedules, data=self.data)
        for i, each_element in enumerate(self.X):
            self.X[i] = each_element + list(range(20))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = NNSmall().to(device)

        print(self.model.state_dict())
        params = list(self.model.parameters())
        self.opt = torch.optim.Adam(params)
        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.convergence_epsilon = .01
        self.when_to_save = 1000
        self.distribution_epsilon = .0001

    def train(self):
        """
        Trains BDT.
        Randomly samples a schedule and timestep within that schedule, and passes in the corresponding data in an attempt to classify which task was scheduled
        :return:
        """

        loss_func = AlphaLoss()
        training_done = False

        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(len(self.X))
            input_nn = self.X[rand_timestep_within_sched]
            truth_nn = self.Y[rand_timestep_within_sched]

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
            if loss.item() < .001 or loss.item() > 30:
                pass
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.opt.step()

            self.total_loss_array.append(loss.item())

            total_iterations = len(self.total_loss_array)

            if total_iterations % 50 == 49:
                print('loss at', total_iterations, ', total loss (average for each 40, averaged)', np.mean(self.total_loss_array[-40:]))

            if total_iterations > 25000:
                training_done = True


    def evaluate_on_test_data(self):

        """
        Evaluate performance of a trained network.
        This is tested on 20% of the data and will be stored in a text file.
        :return:
        """

        num_schedules = 75
        # load in new data
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/test/' + str(
            num_schedules) + '_inf_hetero_deadline_naive.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_dataset(data, num_schedules)
        for i, each_element in enumerate(X):
            X[i] = each_element + list(range(20))


        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        for i, schedule in enumerate(schedule_array):

            for count in range(schedule[0], schedule[1] + 1):

                net_input = X[count]
                truth = Y[count]

                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)))
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)))

                #####forward#####
                output = self.model.forward(input_nn)

                index = torch.argmax(output).item()

                # top 3
                _, top_three = torch.topk(output, 3)

                if index == truth.item():
                    prediction_accuracy[0] += 1

                if truth.item() in top_three.detach().cpu().tolist()[0]:
                    prediction_accuracy[1] += 1


            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)


            print('schedule num:', i)


            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)
            prediction_accuracy = [0, 0]

        print(np.mean(percentage_accuracy_top1))
        self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'inf_nn_small_' + str(self.num_schedules))

    def save_trained_nets(self, name):
        """
        saves the model
        :return:
        """
        torch.save({'nn_state_dict': self.model.state_dict(),
                    'parameters': self.arguments},
                   '/home/ghost/PycharmProjects/bayesian_prolo/saved_models/naive_saved_models/NN_' + name + '.tar')

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
        trainer = NNTrain(num_schedules)
        trainer.train()
        trainer.evaluate_on_test_data()


if __name__ == '__main__':
    main()
