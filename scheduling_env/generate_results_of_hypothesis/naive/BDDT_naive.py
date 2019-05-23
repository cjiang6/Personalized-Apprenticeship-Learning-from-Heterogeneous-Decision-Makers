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
from utils.naive_utils import create_new_dataset, find_which_schedule_this_belongs_to
from Ghost.tree_nets.utils.deepen_prolo_supervised import deepen_with_embeddings

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
            self.num_schedules) + '_inf_hetero_deadline_naive.pkl'

        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_dataset(num_schedules=self.num_schedules, data=self.data)
        for i, each_element in enumerate(self.X):
            self.X[i] = each_element + list(range(20))

        self.model = ProLoNet(input_dim=len(self.X[0]),
                              weights=None,
                              comparators=None,
                              leaves=16,
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
        params = list(self.model.parameters())
        del params[0]
        self.opt = torch.optim.RMSprop([{'params': params}, {'params': self.model.bayesian_embedding, 'lr': .001}])

        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.covergence_epsilon = .01
        self.when_to_save = 1000
        self.distribution_epsilon = .0001

        self.max_depth = 10  # TODO: add back in deepening
        self.embedding_list = [torch.ones(8) * 1 / 3 for _ in range(self.num_schedules)]

    def train(self):
        """
        Trains PDDT.
        :return:
        """

        threshold = .05
        training_done = False
        loss_func = AlphaLoss()

        # deepening data
        deepen_data = {
            'samples': [],
            'labels': [],
            'embedding_indices': []
        }



        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(len(self.X))
            input_nn = self.X[rand_timestep_within_sched]
            truth_nn = self.Y[rand_timestep_within_sched]

            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array, rand_timestep_within_sched)
            load_in_embedding(self.model, self.embedding_list, which_schedule)

            deepen_data['samples'].append(np.array(input_nn))
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

            deepen_data['labels'].extend([truth.item()])
            deepen_data['embedding_indices'].extend([which_schedule])

            # add average loss to array
            # print(list(self.model.parameters()))

            self.embedding_list = store_embedding_back(self.model, self.embedding_list, which_schedule)

            self.total_loss_array.append(loss.item())
            self.total_iterations += 1

            if self.total_iterations > 25 and self.total_iterations % 50 == 1:
                print('total iterations is', self.total_iterations)
                print('total loss (average for each 40, averaged)', np.mean(self.total_loss_array[-40:]))

            if self.total_iterations > 0 and self.total_iterations % self.when_to_save == self.when_to_save - 1:
                self.save_trained_nets('BDDT' + str(self.num_schedules))
                threshold -= .1

            if self.total_iterations % 500 == 499:
                self.model = deepen_with_embeddings(self.model, deepen_data, self.embedding_list, max_depth=self.max_depth,
                                                    threshold=threshold / len(self.model.leaf_init_information))
                params = list(self.model.parameters())
                del params[0]
                self.opt = torch.optim.RMSprop([{'params': params}, {'params': self.model.bayesian_embedding, 'lr': .001}])
                deepen_data = {
                    'samples': [],
                    'labels': [],
                    'embedding_indices': []
                }

            if self.total_iterations > 5000 and np.mean(self.total_loss_array[-100:]) - np.mean(
                    self.total_loss_array[-500:]) < self.covergence_epsilon:
                training_done = True

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
            num_schedules) + '_inf_hetero_deadline_naive.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_dataset(num_schedules=num_schedules, data=data)
        for i, each_element in enumerate(X):
            X[i] = each_element + list(range(20))

        embedding_optimizer = torch.optim.RMSprop([{'params': self.model.bayesian_embedding, 'lr': .001}])
        embedding_list = [torch.ones(8) * 1 / 3 for _ in range(num_schedules)]

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        if load_in_model:
            self.model.load_state_dict(torch.load('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/model_homog.tar')['nn_state_dict'])

        for i, schedule in enumerate(schedule_array):
            load_in_embedding(self.model, embedding_list, i)
            for count in range(schedule[0], schedule[1] + 1):

                net_input = X[count]
                truth = Y[count]

                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)).cuda())  # change to 5 to increase batch size
                    P = Variable(torch.Tensor(np.ones((1, 20)))).cuda()
                    P *= self.distribution_epsilon
                    P[0][truth] = 1 - 19 * self.distribution_epsilon
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)))
                    P = Variable(torch.Tensor(np.ones((1, 20) * self.distribution_epsilon)))
                    P[0][truth] = 1 - 19 * self.distribution_epsilon
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).long())

                #####forward#####
                output = self.model.forward(input_nn)
                embedding_optimizer.zero_grad()
                loss = loss_func.forward(P, output, self.alpha)
                if loss.item() < .001 or loss.item() > 30:
                    pass
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    embedding_optimizer.step()

                index = torch.argmax(output).item()

                # top 3
                _, top_three = torch.topk(output, 3)

                if index == truth.item():
                    prediction_accuracy[0] += 1

                if truth.item() in top_three.detach().cpu().tolist()[0]:
                    prediction_accuracy[1] += 1

            # add average loss to array
            embedding_list = store_embedding_back(self.model, embedding_list, i)

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', i)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'inf_BDT' + str(self.num_schedules))

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
    for num_schedules in (3, 9, 15, 150, 1500):
        trainer = ProLoTrain(num_schedules)
        trainer.train()
        trainer.evaluate_on_test_data()


if __name__ == '__main__':
    main()
