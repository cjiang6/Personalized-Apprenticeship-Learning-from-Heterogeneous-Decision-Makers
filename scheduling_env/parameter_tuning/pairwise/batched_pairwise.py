"""
File to test several alpha values with the goal of finding the optimal divergence for learning a
# scheduling policy
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
import glob
from utils.global_utils import save_pickle, load_in_embedding, store_embedding_back
import matplotlib.pyplot as plt
from utils.pairwise_utils import create_new_data

sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(999)
np.random.seed(1)



# TODO: put data generators into a seperate file since they will need to be called in every naive and pairwise file
# TODO: check how important batching is, maybe parallelism is better?
# TODO: import functions from pairwise_utils


class BDTTrain:
    """
    class structure to train the BDT with a certain alpha.
    This class handles training the BDT, evaluating the BDT, and saving
    """

    def __init__(self, alpha):
        self.arguments = Logger()
        self.alpha = alpha
        self.num_schedules = 150  
        self.home_dir = self.arguments.home_dir
        self.total_loss_array = []

        load_directory = '/home/ghost/PycharmProjects/scheduling_environment/new_data_pickle/' + str(
            self.num_schedules) + 'pairwise.pkl'

        self.X = None
        self.Y = None
        self.schedule_array = None
        bayesian_embedding_dim = 14
        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_data(self.num_schedules, self.data)
        self.start_of_each_set_twenty = self.create_sets_of_20_from_x_for_pairwise_comparisions()
        self.embedding_list = [torch.ones(bayesian_embedding_dim) * 1 / 3 for _ in range(self.num_schedules)]

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = ProLoNet(input_dim=len(self.X[0]),
                              weights=None,
                              comparators=None,
                              leaves=16,
                              output_dim=2,
                              bayesian_embedding_dim=bayesian_embedding_dim,
                              alpha=1.5,
                              use_gpu=True,
                              vectorized=True,
                              is_value=False).cuda()

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




    def create_sets_of_20_from_x_for_pairwise_comparisions(self):
        """
        Create sets of 20 to denote each timestep for all schedules
        :return: range(0, length_of_X, 20)
        """
        length_of_X = len(self.X)
        return list(range(0, length_of_X, 20))

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

    def train(self):
        """
        Trains BDT.
        Randomly samples a schedule and timestep within that schedule, produces training data using x_i - x_j
        and trains upon that.
        :return:
        """
        # loss = nn.CrossEntropyLoss()

        training_done = False
        cv_cutoff = .8 * len(self.start_of_each_set_twenty)
        loss_func = AlphaLoss()

        # variables to keep track of loss and number of tasks trained over
        running_loss_predict_tasks = 0
        num_iterations_predict_task = 0
        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(cv_cutoff)
            set_of_twenty = self.start_of_each_set_twenty[rand_timestep_within_sched]
            truth = self.Y[set_of_twenty]

            which_schedule = self.find_which_schedule_this_belongs_to(set_of_twenty)
            load_in_embedding(self.model, self.embedding_list, which_schedule)

            batched_set = []
            batched_output = []
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
                    batched_set.append(feature_input)
                    # label = add_noise_pairwise(label, self.noise_percentage)
                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)).cuda())
                        P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)))
                        P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]))
                    batched_output.append(P)




            output = self.model(torch.Tensor(batched_set).cuda())
            loss = loss_func.forward(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]).expand(19,2).view(19,1,2).cuda(), output, self.alpha)
            if torch.isnan(loss):
                print(self.alpha, ' :nan occurred at iteration ', self.total_iterations)
                return
                # TODO: can prob just set training to true here if network still outputs proper stuff
            # prepare optimizer, compute gradient, update params
            self.opt.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.opt.step()

            running_loss_predict_tasks += loss.item()
            print('after ', num_iterations_predict_task, ' the embedding has changed to :', self.model.state_dict()['bayesian_embedding'])
            num_iterations_predict_task += 1

            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_j_numpy - phi_i_numpy
                    batched_set.append(feature_input)

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)).cuda())
                        P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)))
                        P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]))

            output = self.model(torch.Tensor(batched_set).cuda())
            loss = loss_func.forward(torch.Tensor([self.distribution_epsilon,1- self.distribution_epsilon]).expand(19, 2).view(19, 1, 2).cuda(), output, self.alpha)

            # if num_iterations_predict_task % 5 == 0:
            #     print('loss is :', loss.item())
            # clip any very high gradients

            # prepare optimizer, compute gradient, update params
            self.opt.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.opt.step()
            print('after ', num_iterations_predict_task, ' the embedding has changed to :', self.model.state_dict()['bayesian_embedding'])
            running_loss_predict_tasks += loss.item()
            num_iterations_predict_task += 1

            # add average loss to array
            # print(list(self.model.parameters()))
            store_embedding_back(self.model, self.embedding_list, which_schedule)
            self.total_loss_array.append(running_loss_predict_tasks / num_iterations_predict_task)
            num_iterations_predict_task = 0
            running_loss_predict_tasks = 0

            self.total_iterations += 1


            if self.total_iterations > 25 and self.total_iterations % 50 == 1:
                print('total iterations is', self.total_iterations)
                print('total loss (average for each 40, averaged)', np.mean(self.total_loss_array[-40:]))
                # TODO: change running loss to actual loss

            if self.total_iterations > 0 and self.total_iterations % self.when_to_save == self.when_to_save - 1:
                # self.plot_nn()
                print(self.embedding_list)
                self.save_trained_nets()
                # self.evaluate()

            if self.total_iterations > 500000 and np.mean(self.total_loss_array[-100:]) - np.mean(
                    self.total_loss_array[-500:]) < self.covergence_epsilon:
                training_done = True

    def evaluate(self, load_in_model=False):
        # TODO: can be changed to one batched forward pass
        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        This is tested on 20% of the data and will be stored in a text file.
        Note this function is called after training convergence
        :return:
        """
        # define new optimizer that only optimizes gradient
        loss_func = AlphaLoss()
        checkpoint = self.model.state_dict().copy()
        optimizer_for_embedding = self.opt = torch.optim.RMSprop([{'params': self.model.bayesian_embedding, 'lr': .001}])
        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        if load_in_model:
            self.model.load_state_dict(torch.load('/home/ghost/PycharmProjects/bayesian_prolo/model.tar')['nn_state_dict'])
        # for rest of schedule
        # i = .8 * len(self.start_of_each_set_twenty)
        # num_test_schedules = 150 * .2
        for j in range(int(self.num_schedules * .8), self.num_schedules):
            load_in_embedding(self.model, self.embedding_list, j)
            schedule_bounds = self.schedule_array[j]
            step = schedule_bounds[0]
            while step < schedule_bounds[1]:
                probability_matrix = np.zeros((20, 20))

                for m, counter in enumerate(range(step, step + 20)):
                    phi_i = self.X[counter]
                    phi_i_numpy = np.asarray(phi_i)

                    # for each set of twenty
                    for n, second_counter in enumerate(range(step, step + 20)):
                        # fill entire array with diagnols set to zero
                        if second_counter == counter:  # same as m = n
                            continue
                        phi_j = self.X[second_counter]
                        phi_j_numpy = np.asarray(phi_j)

                        feature_input = phi_i_numpy - phi_j_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)).cuda())

                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)))

                        # push through nets
                        preference_prob = self.model.forward(feature_input)
                        probability_matrix[m][n] = preference_prob[0].data.detach()[0].item()
                        probability_matrix[n][m] = preference_prob[0].data.detach()[1].item()

                # Set of twenty is completed
                column_vec = np.sum(probability_matrix, axis=1)

                # top 1
                choice = np.argmax(column_vec)

                # top 3
                _, top_three = torch.topk(torch.Tensor(column_vec), 3)

                # Then do training update loop
                truth = self.Y[step]

                # index top 1
                if choice == truth:
                    prediction_accuracy[0] += 1

                # index top 3
                if truth in top_three:
                    prediction_accuracy[1] += 1

                # forward
                phi_i_num = truth + step  # old method: set_of_twenty[0] + truth
                phi_i = self.X[phi_i_num]
                phi_i_numpy = np.asarray(phi_i)
                # iterate over pairwise comparisons
                for counter in range(step, step + 20):
                    if counter == phi_i_num:  # if counter == phi_i_num:
                        continue
                    else:
                        phi_j = self.X[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_i_numpy - phi_j_numpy

                        # label = add_noise_pairwise(label, self.noise_percentage)
                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)).cuda())
                            P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)))
                            P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]))

                        output = self.model(feature_input)
                        loss = loss_func.forward(P, output, self.alpha)
                        # prepare optimizer, compute gradient, update params
                        optimizer_for_embedding.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        optimizer_for_embedding.step()

                for counter in range(step, step + 20):
                    if counter == phi_i_num:
                        continue
                    else:
                        phi_j = self.X[counter]
                        phi_j_numpy = np.asarray(phi_j)
                        feature_input = phi_j_numpy - phi_i_numpy

                        if torch.cuda.is_available():
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)).cuda())
                            P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]).cuda())
                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)))
                            P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]))

                        output = self.model(feature_input)
                        loss = loss_func.forward(P, output, self.alpha)
                        # print('loss is :', loss.item())
                        # clip any very high gradients

                        # prepare optimizer, compute gradient, update params
                        self.opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.opt.step()

                # add average loss to array
                store_embedding_back(self.model, self.embedding_list, j)
                step += 20

            # schedule finished
            print('Prediction Accuracy: top1: ', prediction_accuracy[0] / 20, ' top3: ', prediction_accuracy[1] / 20)

            print('schedule num:', j)
            percentage_accuracy_top1.append(prediction_accuracy[0] / 20)
            percentage_accuracy_top3.append(prediction_accuracy[1] / 20)

            prediction_accuracy = [0, 0]
        self.save_performance_results(percentage_accuracy_top1, percentage_accuracy_top3)
        self.model.load_state_dict(checkpoint)

    def save_trained_nets(self):
        """
        saves the model
        :return:
        """
        torch.save({'nn_state_dict': self.model.state_dict(),
                    'parameters': self.arguments},
                   '/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/model' + str(self.alpha) + '.tar')

    def save_performance_results(self, top1, top3):
        """
        saves performance of top1 and top3
        :return:
        """
        print('top1_mean for ', self.alpha, ' is : ', np.mean(top1))
        data = {'top1_mean': np.mean(top1),
                'top3_mean': np.mean(top3),
                'top1_stderr': np.std(top1) / np.sqrt(len(top1)),
                'top3_stderr': np.std(top3) / np.sqrt(len(top3)),
                'alpha': self.alpha}
        save_pickle(file=data, file_location=self.home_dir + '/saved_models/pairwise_saved_models/', special_string=str(self.alpha) + 'alpha.pkl')
        if np.mean(top1) > .6:
            exit()

    # TODO: function that searches through all the saved means and stderr, plots, and finds the highest

    def find_optimal_alpha(self):
        """
        Searches through all .pkl files and returns the name of the file with the highest top 1 accuracy
        :return:
        """
        performance_files = glob.glob(self.home_dir + '/*.pkl')
        performance_files2 = glob.glob(self.home_dir + '/saved_models/pairwise_saved_models/*alpha.pkl')
        print(performance_files2)
        performance_files = performance_files + performance_files2
        max_val = 0
        max_filename = None
        alpha = []
        vals= []
        for i, file in enumerate(performance_files):
            data: dict = pickle.load(open(file, 'rb'))
            print(file, ': ', data['top1_mean'])
            if data['top1_mean'] > max_val:
                max_val = data['top1_mean']
                max_filename = file
            if 'alpha' in data.keys():
                alpha.append(data['alpha'])
                vals.append(data['top1_mean'])

        plt.scatter(alpha,vals)
        plt.show()

        print('The optimal alpha is: ', max_filename, 'with an accuracy of: ', max_val)
        print('The exact value of alpha can be found by dividing the digits before the .tar by 10')

    # TODO: add k-fold cross validation


def main():
    """
    entry point for file
    :return:
    """
    trainer = None
    # for alpha in np.linspace(.01, 3, num=30):
    #     trainer = BDTTrain(alpha)
    #     trainer.train()
    #     trainer.evaluate()

    trainer = BDTTrain(.9)
    trainer.train()
    trainer.evaluate()
    trainer.find_optimal_alpha()


if __name__ == '__main__':
    main()
