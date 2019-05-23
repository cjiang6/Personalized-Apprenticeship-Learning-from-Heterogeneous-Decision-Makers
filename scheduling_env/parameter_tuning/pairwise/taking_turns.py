"""
File to test several embedding values with the goal of finding the optimal divergence for learning a
# scheduling policy
"""

import torch
import sys
sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')
from scheduling_env.alpha_div import AlphaLoss
from Ghost.tree_nets.vectorized_prolonet import ProLoNet
import numpy as np
from scheduling_env.argument_parser import Logger
import pickle
from torch.autograd import Variable
import glob
from utils.global_utils import save_pickle, load_in_embedding, store_embedding_back
# import matplotlib.pyplot as plt
from utils.pairwise_utils import create_new_data

sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)


# noinspection PyTypeChecker
class BDTTrain:
    """
    class structure to train the BDT with a certain embedding.
    This class handles training the BDT, evaluating the BDT, and saving
    """

    def __init__(self, bayesian_dim):
        self.arguments = Logger()
        self.alpha = .9
        self.num_schedules = 200  
        self.home_dir = self.arguments.home_dir
        self.total_loss_array = []
        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + '_BDFIL_hetero_deadline_pairwise.pkl'

        self.bayesian_embedding_dim = int(bayesian_dim)
        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_data(self.num_schedules, self.data)
        self.start_of_each_set_twenty = self.create_sets_of_20_from_x_for_pairwise_comparisions()
        self.embedding_list = [torch.ones(self.bayesian_embedding_dim) * 1 / 3 for _ in range(self.num_schedules)]

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        use_gpu = True
        self.model = ProLoNet(input_dim=len(self.X[0]),
                              weights=None,
                              comparators=None,
                              leaves=16,
                              output_dim=2,
                              bayesian_embedding_dim=self.bayesian_embedding_dim,
                              alpha=1.5,
                              use_gpu=use_gpu,
                              vectorized=True,
                              is_value=False).cuda()

        if use_gpu:
            self.model = self.model.cuda()
        print(self.model.state_dict())
        params = list(self.model.parameters())
        del params[0]
        self.opt = torch.optim.RMSprop(params, lr=.0001)
        self.embedding_optimizer = torch.optim.Adam([{'params': self.model.bayesian_embedding, 'lr': .01}])
        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.covergence_epsilon = .01
        self.when_to_save = 2000
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

            # get the current schedule that the timestep is from
            which_schedule = self.find_which_schedule_this_belongs_to(set_of_twenty)

            # load in the embedding from the list into the network
            load_in_embedding(self.model, self.embedding_list, which_schedule)

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

                    # label = add_noise_pairwise(label, self.noise_percentage)
                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]))

                    output = self.model(feature_input)
                    loss = loss_func.forward(P, output, self.alpha)



                    # prepare optimizer, compute gradient, update params
                    if self.total_iterations % 2000 > 1000:
                        self.embedding_optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.embedding_optimizer.step()
                    else:
                        self.opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.opt.step()

                    running_loss_predict_tasks += loss.item()
                    num_iterations_predict_task += 1

            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_j_numpy - phi_i_numpy

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                        P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]).cuda())
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                        P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]))

                    output = self.model(feature_input)
                    loss = loss_func.forward(P, output, self.alpha)


                    # prepare optimizer, compute gradient, update params
                    if self.total_iterations % 2000 > 1000:
                        self.embedding_optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.embedding_optimizer.step()
                    else:
                        self.opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.opt.step()

                    running_loss_predict_tasks += loss.item()
                    num_iterations_predict_task += 1

            self.embedding_list=store_embedding_back(self.model, self.embedding_list, which_schedule)
            self.total_loss_array.append(running_loss_predict_tasks / num_iterations_predict_task)
            num_iterations_predict_task = 0
            running_loss_predict_tasks = 0

            self.total_iterations += 1

            if self.total_iterations > 25 and self.total_iterations % 50 == 1:
                print('total iterations is', self.total_iterations)
                print('total loss (average for each 40, averaged)', np.mean(self.total_loss_array[-40:]))

            if self.total_iterations > 0 and self.total_iterations % self.when_to_save == self.when_to_save - 1:
                print(self.embedding_list)
                self.save_trained_nets()

            if self.total_iterations > 8000 and np.mean(self.total_loss_array[-100:]) - np.mean(
                    self.total_loss_array[-500:]) < self.covergence_epsilon:
                training_done = True

    def evaluate(self, load_in_model=False):

        """
        Evaluate performance of a trained network tuned upon the alpha divergence loss.
        This is tested on 20% of the data and will be stored in a text file.
        Note this function is called after training convergence
        :return:
        """
        # define new optimizer that only optimizes gradient
        loss_func = AlphaLoss()
        checkpoint = self.model.state_dict().copy()
        embedding_list_copy = self.embedding_list.copy()
        prediction_accuracy = [0, 0]
        percentage_accuracy_top1 = []
        percentage_accuracy_top3 = []

        if load_in_model:
            self.model.load_state_dict(torch.load('/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/model_baydimtest_' + str(self.bayesian_embedding_dim) + '.tar')['nn_state_dict'])
        # for rest of schedule
        # i = .8 * len(self.start_of_each_set_twenty)
        # num_test_schedules = 150 * .2
        for j in range(int(self.num_schedules * .8), self.num_schedules):
            optimizer_for_embedding = self.opt = torch.optim.SGD([{'params': self.model.bayesian_embedding, 'lr': .5}])
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
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())

                        else:
                            feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))

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
                if choice == truth:
                    pass
                else:
                    for counter in range(step, step + 20):
                        if counter == phi_i_num:  # if counter == phi_i_num:
                            continue
                        else:
                            phi_j = self.X[counter]
                            phi_j_numpy = np.asarray(phi_j)
                            feature_input = phi_i_numpy - phi_j_numpy

                            # label = add_noise_pairwise(label, self.noise_percentage)
                            if torch.cuda.is_available():
                                feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                                P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]).cuda())
                            else:
                                feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                                P = Variable(torch.Tensor([1 - self.distribution_epsilon, self.distribution_epsilon]))

                            output = self.model(feature_input)
                            loss = loss_func.forward(P, output, self.alpha)
                            # prepare optimizer, compute gradient, update params
                            if loss.item() > .005:
                                flag = False
                                tracker = 0
                                while not flag:
                                    optimizer_for_embedding.zero_grad()
                                    loss.backward()
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                                    optimizer_for_embedding.step()
                                    output = self.model(feature_input)
                                    loss = loss_func.forward(P, output, self.alpha)
                                    tracker += 1
                                    if tracker > 50:
                                        flag = True
                                    if loss.item() < .005:
                                        flag = True

                            else:
                                pass

                    for counter in range(step, step + 20):
                        if counter == phi_i_num:
                            continue
                        else:
                            phi_j = self.X[counter]
                            phi_j_numpy = np.asarray(phi_j)
                            feature_input = phi_j_numpy - phi_i_numpy

                            if torch.cuda.is_available():
                                feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)).cuda())
                                P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]).cuda())
                            else:
                                feature_input = Variable(torch.Tensor(feature_input.reshape(1, 13)))
                                P = Variable(torch.Tensor([self.distribution_epsilon, 1 - self.distribution_epsilon]))

                            output = self.model(feature_input)
                            loss = loss_func.forward(P, output, self.alpha)


                            # prepare optimizer, compute gradient, update params
                            if loss.item() > .005:
                                flag = False
                                tracker = 0
                                while not flag:
                                    optimizer_for_embedding.zero_grad()
                                    loss.backward()
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                                    optimizer_for_embedding.step()
                                    output = self.model(feature_input)
                                    loss = loss_func.forward(P, output, self.alpha)
                                    tracker += 1
                                    if tracker > 50:
                                        flag = True
                                    if loss.item() < .005:
                                        flag = True

                            else:
                                pass

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
        self.embedding_list = embedding_list_copy

    def save_trained_nets(self):
        """
        saves the model
        :return:
        """
        torch.save({'nn_state_dict': self.model.state_dict(),
                    'parameters': self.arguments},
                   '/home/ghost/PycharmProjects/bayesian_prolo/saved_models/pairwise_saved_models/model_baydimtest_' + str(self.bayesian_embedding_dim) + '.tar')

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
        save_pickle(file=data, file_location=self.home_dir + '/saved_models/pairwise_saved_models/', special_string=str(self.bayesian_embedding_dim) + 'baydimtest.pkl')


    def find_optimal_embedding(self):
        """
        Searches through all .pkl files and returns the name of the file with the highest top 1 accuracy
        :return:
        """

        performance_files = glob.glob(self.home_dir + '/saved_models/pairwise_saved_models/*test.pkl')
        print(performance_files)
        performance_files = performance_files

        for i, file in enumerate(performance_files):
            data: dict = pickle.load(open(file, 'rb'))
            print(file, ': ', data['top1_mean'])

        # plt.scatter(alpha, vals)
        # plt.show()


def main():
    """
    entry point for file
    :return:
    """
    trainer = None
    for bayesian_embedding_dim in np.linspace(1, 16, num=16):
        pass

    trainer = BDTTrain(1)
    trainer.train()
    trainer.evaluate()




if __name__ == '__main__':
    main()
