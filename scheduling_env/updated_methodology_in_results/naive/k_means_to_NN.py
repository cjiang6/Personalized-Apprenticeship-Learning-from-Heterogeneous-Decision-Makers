"""
Testing the NN_small. This is expected to do much worse than the BDDT
"""

import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

# sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')
from scheduling_env.alpha_div import AlphaLoss
import numpy as np
from scheduling_env.argument_parser import Logger
import pickle
from torch.autograd import Variable
from utils.global_utils import save_pickle
from utils.naive_utils import create_new_dataset, find_which_schedule_this_belongs_to
from sklearn.cluster import KMeans
from scheduling_env.generate_results_of_hypothesis.pairwise.train_autoencoder import Autoencoder, AutoEncoderTrain
# sys.path.insert(0, '../')
import itertools
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)
np.random.seed(50)


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
        self.soft = nn.LogSoftmax(dim=1)

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

    def __init__(self):
        self.arguments = Logger()
        self.alpha = .9
        self.num_schedules = 150  
        self.home_dir = self.arguments.home_dir
        self.total_loss_array = []

        # TODO: load in new directory
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + 'dist_early_hili_naive.pkl'

        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_dataset(num_schedules=self.num_schedules, data=self.data)
        for i, each_element in enumerate(self.X):
            self.X[i] = each_element + list(range(20))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model1 = NNSmall().to(device)
        model2 = NNSmall().to(device)
        model3 = NNSmall().to(device)

        self.models = [model1, model2, model3]

        opt1 = torch.optim.RMSprop(model1.parameters(), lr=.0001)
        opt2 = torch.optim.RMSprop(model2.parameters(), lr=.0001)
        opt3 = torch.optim.RMSprop(model3.parameters(), lr=.0001)

        self.optimizers = [opt1, opt2, opt3]
        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.convergence_epsilon = .01
        self.when_to_save = 1000
        self.distribution_epsilon = .0001


        schedule_matrix_load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/' + str(self.num_schedules) + 'matrixes.pkl'
        self.matrices = pickle.load(open(schedule_matrix_load_directory, "rb"))

        self.kmeans_model, self.label = self.cluster_matrices(self.matrices, self.num_schedules)


    @staticmethod
    def cluster_matrices(matrices, num_schedules):
        # vectorize each matrix
        vectorized_set = []
        for i in matrices:
            vectorized = i.reshape(20 * 2048, 1)
            vectorized_set.append(vectorized)
        kmeans = KMeans(n_clusters=3, random_state=0) # random state makes it deterministic
        # Fitting the input data
        new_set = np.hstack(tuple(vectorized_set)).reshape(num_schedules, 20 * 2048)
        kmeans_model = kmeans.fit(np.asarray(new_set))
        labels = kmeans_model.predict(np.asarray(new_set))
        return kmeans_model, labels


    # noinspection PyTypeChecker
    def train(self):
        """
        Trains NN.
        Randomly samples a schedule and timestep within that schedule, and passes in the corresponding data in an attempt to classify which task was scheduled
        :return:
        """


        training_done = False
        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(len(self.X))
            input_nn = self.X[rand_timestep_within_sched]
            truth_nn = self.Y[rand_timestep_within_sched]
            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array, rand_timestep_within_sched)
            cluster_num = self.label[which_schedule]
            # iterate over pairwise comparisons
            if torch.cuda.is_available():
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda())  # change to 5 to increase batch size
                truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
            else:
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))
                truth = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).long())

            self.optimizers[cluster_num].zero_grad()
            output = self.models[cluster_num].forward(input_nn)
            loss = F.cross_entropy(output, truth)

            # TODO: check if this loop is still needed
            # print(self.models[cluster_num].state_dict())
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizers[cluster_num].step()

            self.total_loss_array.append(loss.item())

            total_iterations = len(self.total_loss_array)

            if total_iterations % 500 == 499:
                print('loss at', total_iterations, ', total loss (average for each 40, averaged)', np.mean(self.total_loss_array[-40:]))


            if total_iterations > 25000:
                training_done = True

    def create_iterables(self):
        """
        adds all possible state combinations
        :return:
        """
        iterables = [[0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1], [0, 1],
                     [0, 1], [0, 1]]
        states = []
        for t in itertools.product(*iterables):
            states.append(t)
        return states

    def pass_in_embedding_out_state_ID(self, states, binary):
        """
        pass in a binary embedding, and itll return the state id
        :param binary:
        :return:
        """
        binary_as_tuple = tuple(binary)
        index = states.index(binary_as_tuple)
        return index


    def evaluate_on_test_data(self):

        """
        Evaluate performance of a trained network.
        This is tested on 20% of the data and will be stored in a text file.
        :return:
        """
        # confusion_matrix = np.zeros((20,20))
        num_schedules = 100

        autoencoder_class = AutoEncoderTrain(150)
        checkpoint = torch.load('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/Autoencoder150.tar')
        autoencoder_class.model.load_state_dict(checkpoint['nn_state_dict'])
        states = self.create_iterables()
        # load in new data

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            num_schedules) + 'test_dist_early_hili_naive.pkl'

        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_dataset(data, num_schedules)
        for i, each_element in enumerate(X):
            X[i] = each_element + list(range(20))

        # only use last 100 as held out dataset
        # X = X[-2000:]
        # Y = Y[-2000:]


        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []
        mean_input = [1.3277743 , 0.32837677 ,1.4974482, -1.3519306 ,-0.64621973 , 0.10534518, -2.338118 ,-2.7345326  , 1.7558736, -3.0746384, -3.485554]
        for i, schedule in enumerate(schedule_array):
            current_schedule_matrix = np.zeros((2048, 20))

            for count in range(schedule[0], schedule[1] + 1):
                if current_schedule_matrix.sum() == 0:
                    cluster_num = self.kmeans_model.predict(current_schedule_matrix.reshape(1, -1))
                else:
                    matrix = np.divide(current_schedule_matrix, current_schedule_matrix.sum())
                    cluster_num = self.kmeans_model.predict(matrix.reshape(1, -1))


                net_input = X[count]
                truth = Y[count]

                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)))
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)))

                #####forward#####
                output = self.models[int(cluster_num)].forward(input_nn)

                index = torch.argmax(output).item()

                # confusion_matrix[truth][index] += 1
                # top 3
                _, top_three = torch.topk(output, 3)

                if index == truth.item():
                    prediction_accuracy[0] += 1

                if truth.item() in top_three.detach().cpu().tolist()[0]:
                    prediction_accuracy[1] += 1

                    # update matrix

                embedding_copy = np.zeros((1, 11))
                input_element = autoencoder_class.model.forward_only_encoding(input_nn)
                for z, each_element in enumerate(mean_input):
                    if each_element > input_element[0][z].item():
                        embedding_copy[0][z] = 0
                    else:
                        embedding_copy[0][z] = 1
                index = self.pass_in_embedding_out_state_ID(states, embedding_copy[0])
                action = truth.item()
                current_schedule_matrix[index][int(action)] += 1


            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', i)

            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)
            prediction_accuracy = [0, 0]

        print(np.mean(percentage_accuracy_top1))
        self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'kmeans_to_NN_naive')


        return percentage_accuracy_top1


    def save_trained_nets(self, name):
        """
        saves the model
        :return:
        """
        torch.save({'nn1_state_dict': self.models[0].state_dict(),
                    'nn2_state_dict': self.models[1].state_dict(),
                    'nn3_state_dict': self.models[2].state_dict(),
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

    trainer = NNTrain()
    trainer.train()
    per_schedule_test_accs = trainer.evaluate_on_test_data()

    file = open('scheduling_env_results.txt', 'a')
    file.write('k_means NN: mean: ' +
               str(np.mean(per_schedule_test_accs)) +
               ', std: ' + str(np.std(per_schedule_test_accs)) +
               '\n')
    file.close()

if __name__ == '__main__':
    main()
