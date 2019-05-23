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
from utils.global_utils import save_pickle, load_in_embedding_bnn, store_embedding_back_bnn
from utils.naive_utils import create_new_dataset, find_which_schedule_this_belongs_to
import copy
sys.path.insert(0, '../')
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)
np.random.seed(50)


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


# noinspection PyTypeChecker
class NNwEmbedding(nn.Module):
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
        super(NNwEmbedding, self).__init__()
        self.fc1 = nn.Linear(245, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(128, 128)
        self.relu21 = nn.ReLU()
        self.fc22 = nn.Linear(128, 32)
        self.relu22 = nn.ReLU()
        self.fc23 = nn.Linear(32, 32)
        self.relu23 = nn.ReLU()
        self.fc3 = nn.Linear(32, 20)
        self.soft = nn.Softmax()
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
        x = self.fc23(x)
        x = self.relu23(x)
        x = self.fc3(x)
        x = self.soft(x)

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

# noinspection PyTypeChecker
class NNwEmbeddingTrain:
    """
    class structure to train the NN for a certain amount of schedules.
    This class handles training the NN, evaluating the NN, and saving the results
    """

    def __init__(self):
        self.arguments = Logger()
        self.alpha = .9
        self.num_schedules = 150  
        self.home_dir = self.arguments.home_dir
        self.total_loss_array = []

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + 'dist_early_hili_naive.pkl'

        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_dataset(num_schedules=self.num_schedules, data=self.data)
        for i, each_element in enumerate(self.X):
            self.X[i] = each_element + list(range(20))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = NNwEmbedding().to(device)

        print(self.model.state_dict())

        self.opt = torch.optim.Adam([{'params': list(self.model.parameters())[:-1]}, {'params': self.model.EmbeddingList.parameters(), 'lr': .01}])
        self.embedding_optimizer = torch.optim.Adam(self.model.EmbeddingList.parameters(), lr=.1)
        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.convergence_epsilon = .01
        self.when_to_save = 1000
        self.distribution_epsilon = .0001

        self.embedding_list = [torch.ones(3) * 1 / 3 for _ in range(self.num_schedules)]

    def train(self):
        """
        Trains BDT.
        Randomly samples a schedule and timestep within that schedule, and passes in the corresponding data in an attempt to classify which task was scheduled
        :return:
        """
        total_iterations = 0
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
            self.opt.step()

            self.total_loss_array.append(loss.item())
            print(self.model.EmbeddingList.state_dict())

            total_iterations = len(self.total_loss_array)
            self.embedding_list[which_schedule] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy()[0])  # very ugly


            if total_iterations % 500== 499:
                print('loss at', total_iterations, ', total loss (average for each 40, averaged)', np.mean(self.total_loss_array[-40:]))

            if total_iterations > 50000:
                training_done = True


    def evaluate_on_test_data(self):

        """
        Evaluate performance of a trained network.
        This is tested on 20% of the data and will be stored in a text file.
        :return:
        """

        num_schedules = 100
        # load in new data
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            num_schedules) + 'test_dist_early_hili_naive.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_dataset(data, num_schedules)
        for i, each_element in enumerate(X):
            X[i] = each_element + list(range(20))


        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []


        embedding_list = [torch.ones(3) * 1/3 for i in range(num_schedules)]

        for i, schedule in enumerate(schedule_array):
            self.model.set_bayesian_embedding(embedding_list[i])
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
                loss = F.cross_entropy(output, truth) * 10
                self.embedding_optimizer.zero_grad()
                print(self.model.EmbeddingList.state_dict())
                loss.backward()
                self.embedding_optimizer.step()
                index = torch.argmax(output).item()
                print(self.model.EmbeddingList.state_dict())
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
        self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'NN_w_embedding')

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

    trainer = NNwEmbeddingTrain()
    trainer.train()
    trainer.evaluate_on_test_data()


if __name__ == '__main__':
    main()
