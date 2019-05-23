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
from utils.pairwise_utils import create_new_data, create_sets_of_20_from_x_for_pairwise_comparisions, save_performance_results, find_which_schedule_this_belongs_to

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
        self.embedding = nn.Parameter(torch.randn(1, 3))

    def forward(self):
        """
        doesn't do anything
        :return:
        """
        return

class NNwEmbedding(nn.Module):
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
        super(NNwEmbedding, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(32, 32)
        self.relu21 = nn.ReLU()
        self.fc22 = nn.Linear(32, 32)
        self.relu22 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()
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
        x = self.fc3(x)
        x = self.sig(x)

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

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + 'dist_early_hili_pairwise.pkl'

        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_data(self.num_schedules, self.data)
        self.start_of_each_set_twenty = create_sets_of_20_from_x_for_pairwise_comparisions(self.X)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = NNwEmbedding().to(device)

        print(self.model.state_dict())
        self.opt = torch.optim.Adam([{'params': list(self.model.parameters())[:-1]}, {'params': self.model.EmbeddingList.parameters(), 'lr':.01}], lr=.001)
        self.embedding_optimizer = torch.optim.SGD(self.model.EmbeddingList.parameters(), lr=.01)

        self.embedding_list = [torch.ones(3) * 1 / 3 for _ in range(self.num_schedules)]


    # noinspection PyArgumentList
    def train(self):
        """
        Trains NN.
        Randomly samples a schedule and timestep within that schedule, produces training data using x_i - x_j
        and trains upon that.
        :return:
        """

        total_iterations = 0
        convergence_epsilon = .01
        when_to_save = 1000
        distribution_epsilon = .0001
        training_done = False
        total_loss_array = []
        criterion = torch.nn.BCELoss()


        # variables to keep track of loss and number of tasks trained over

        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            set_of_twenty = np.random.choice(self.start_of_each_set_twenty)
            truth = self.Y[set_of_twenty]
            which_schedule = find_which_schedule_this_belongs_to(self.schedule_array, set_of_twenty)
            self.model.set_bayesian_embedding(self.embedding_list[which_schedule])

            # find feature vector of true action taken
            phi_i_num = truth + set_of_twenty
            phi_i = self.X[phi_i_num]
            phi_i_numpy = np.asarray(phi_i)
            running_loss_predict_tasks = 0
            num_iterations_predict_task = 0
            # iterate over pairwise comparisons
            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:  # if counter == phi_i_num:
                    # if this is the actual task that was scheduled
                    label = torch.ones((1,1))
                else:
                    label = torch.zeros((1,1))
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


                self.opt.zero_grad()
                loss = criterion(output, label)

                if torch.isnan(loss):
                    print('nan occurred at iteration ', total_iterations, ' at', num_iterations_predict_task)

                if counter == phi_i_num:
                    loss *= 50
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.opt.step()
                # print(self.model.EmbeddingList.state_dict())
                running_loss_predict_tasks += loss.item()
                num_iterations_predict_task += 1


            self.embedding_list[which_schedule] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy()[0])  # very ugly

            total_loss_array.append(running_loss_predict_tasks / num_iterations_predict_task)

            total_iterations += 1

            if total_iterations % 50 == 49:
                print('total loss (average for each 40, averaged) at iteration ', total_iterations, ' is ', np.mean(total_loss_array[-40:]))



            if total_iterations > 5000 and np.mean(total_loss_array[-100:]) - np.mean(total_loss_array[-500:]) < convergence_epsilon:
                training_done = True

    # noinspection PyArgumentList

    def evaluate_on_test_data(self, model):
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        Note this function is called after training convergence
        :return:
        """
        num_schedules = 100
        # load in new data
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            num_schedules) + 'test_dist_early_hili_pairwise.pkl'
        # embedding_optimizer = torch.optim.SGD(model.EmbeddingList.parameters(), lr=.4)

        criterion = torch.nn.BCELoss()


        data = pickle.load(open(load_directory, "rb"))
        X, Y, schedule_array = create_new_data(num_schedules, data)
        embedding_list = [torch.ones(3) * 1 / 3 for i in range(num_schedules)]

        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        for j in range(0, num_schedules):
            schedule_bounds = schedule_array[j]
            step = schedule_bounds[0]
            # for each schedule, set the embedding
            model.set_bayesian_embedding(embedding_list[j])



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
                    probability_vector[0][m] = preference_prob[0].data.detach()[
                        0].item()  # TODO: you can do a check if only this line leads to the same thing as the line below
                        # probability_matrix[n][m] = preference_prob[0].data.detach()[1].item()

                # given all inputs, and their liklihood of being scheduled, predict the output
                highest_val = max(probability_vector[0])
                all_indexes_that_have_highest_val = [i for i, e in enumerate(list(probability_vector[0])) if e == highest_val]
                if len(all_indexes_that_have_highest_val) > 1:
                    print('length of indexes greater than 1: ', all_indexes_that_have_highest_val)
                # top 1
                choice = np.random.choice(all_indexes_that_have_highest_val)
                # choice = np.argmax(probability_vector)

                # top 3
                _, top_three = torch.topk(torch.Tensor(probability_vector), 3)


                truth = Y[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1

                # index top 3
                if truth in top_three:
                    prediction_accuracy[1] += 1

                # update loop
                truth = Y[step]
                phi_i_num = truth + step  # old method: set_of_twenty[0] + truth
                for counter in range(step, step + 20):
                    if counter == phi_i_num:  # if counter == phi_i_num:
                        label = torch.ones((1, 1))
                    else:
                        label = torch.zeros((1, 1))
                    phi_j = X[counter]
                    phi = np.asarray(phi_j)
                    feature_input = phi

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        label = Variable(torch.Tensor(label).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        label = Variable(torch.Tensor(label.reshape(1, 1)))

                    output = model.forward(feature_input)

                    self.embedding_optimizer.zero_grad()
                    loss = criterion(output, label)


                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    self.embedding_optimizer.step()
                    print(model.EmbeddingList.state_dict())
                embedding_list[j] = torch.Tensor(model.get_bayesian_embedding().detach().cpu().numpy()[0])  # very ugly


                # add average loss to array
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3, 'pointwise_NN_unimodal.pkl')

        return percentage_accuracy_top1


    def save_trained_nets(self, name):
        """
        saves the model
        :return:
        """
        torch.save({'nn_state_dict': self.model.state_dict(),
                    'parameters': self.arguments},
                   '/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/NN_' + name + '.tar')


def main():
    """
    entry point for file
    :return:
    """

    num_schedules = 150
    trainer = NNTrain()
    trainer.train()
    per_schedule_test_accs = trainer.evaluate_on_test_data(trainer.model)
    file = open('scheduling_env_results.txt', 'a')
    file.write('Pointwise NN w embedding: mean: ' +
               str(np.mean(per_schedule_test_accs)) +
               ', std: ' + str(np.std(per_schedule_test_accs)) +
               '\n')
    file.close()

if __name__ == '__main__':
    main()
