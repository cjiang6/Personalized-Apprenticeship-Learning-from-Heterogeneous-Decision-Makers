# Pairwise BDT using BCELoss()
import torch
import sys
from Ghost.tree_nets.vectorized_prolonet import ProLoNet
import numpy as np
from scheduling_env.argument_parser import Logger
import ast
import pickle
from torch.autograd import Variable


sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(999)
np.random.seed(1)


def load_in_embedding(NeuralNet, embedding_list, player_id):
    curr_embedding = embedding_list[player_id]
    curr_dict = NeuralNet.state_dict()
    curr_dict['bayesian_embedding'] = curr_embedding
    NeuralNet.load_state_dict(curr_dict)


def store_embedding_back(NeuralNet, embedding_list, player_id, DEBUG=False):
    curr_dict = NeuralNet.state_dict()
    new_embedding = curr_dict['bayesian_embedding'].clone()
    curr_embedding = embedding_list[player_id]
    embedding_list[player_id] = new_embedding
    return embedding_list


class BDT_Train():
    def __init__(self):
        self.arguments = Logger()
        self.num_schedules = 150
        self.home_dir = self.arguments.home_dir
        self.total_loss_array = []

        load_directory = '/home/ghost/PycharmProjects/scheduling_environment/new_data_pickle/' + str(
            self.num_schedules) + 'pairwise.pkl'

        self.X = None
        self.Y = None
        self.schedule_array = None
        bayesian_embedding_dim = 4
        self.data = pickle.load(open(load_directory, "rb"))
        self.create_new_data()
        self.start_of_each_set_twenty = self.create_sets_of_20_from_X_for_pairwise_comparisions()
        self.embedding_list = [torch.ones(bayesian_embedding_dim) * 1 / 3 for i in range(self.num_schedules)]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = ProLoNet(input_dim=len(self.X[0]),
                              weights=None,
                              comparators=None,
                              leaves=64,
                              output_dim=1,
                              bayesian_embedding_dim=bayesian_embedding_dim,
                              alpha=0.5,
                              use_gpu=True,
                              vectorized=False,
                              is_value=True).cuda()

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
        self.when_to_save = 150
        self.criterion = torch.nn.BCELoss()

    def create_new_data(self):
        self.X = []
        self.Y = []
        self.schedule_array = []
        for i in range(0, self.num_schedules):
            timesteps_where_events_are_scheduled = self.find_nums_with_task_scheduled_pkl(i)  # should be 20 sets of 20
            if i == 0:
                start = 0
            else:
                start = self.schedule_array[-1][1] + 1
            end = start + len(timesteps_where_events_are_scheduled) - 1

            self.schedule_array.append([start, end])  # each block is of size 400
            for each_timestep in timesteps_where_events_are_scheduled:
                input_nn, output = self.rebuild_input_output_from_pickle(i, each_timestep)
                self.X.append(input_nn)
                self.Y.append(output)

    def find_nums_with_task_scheduled_pkl(self, rand_schedule):
        nums = []
        for i, timestep in enumerate(self.data[rand_schedule]):
            if ast.literal_eval(self.data[rand_schedule][i][18]) != -1:
                nums.append(i)
            else:
                continue
        return nums

    def rebuild_input_output_from_pickle(self, rand_schedule, rand_timestep):
        schedule_timestep_data = self.data[rand_schedule][rand_timestep]
        state_input = []
        for i, element in enumerate(schedule_timestep_data):
            # if i == 0:
            #     if type(ast.literal_eval(element)) == float:
            #         state_input.append(ast.literal_eval(element))
            #     elif type(ast.literal_eval(element)) == int:
            #         state_input.append(ast.literal_eval(element))
            if i == 17:
                continue

            elif 18 > i > 4:
                if type(ast.literal_eval(element)) == float:
                    state_input.append(ast.literal_eval(element))
                elif type(ast.literal_eval(element)) == int:
                    state_input.append(ast.literal_eval(element))
                elif type(ast.literal_eval(element)) == list:
                    state_input = state_input + ast.literal_eval(element)
            else:
                continue

        output = ast.literal_eval(schedule_timestep_data[18])

        return state_input, output

    def create_sets_of_20_from_X_for_pairwise_comparisions(self):
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
            if sample_val >= each_array[0] and sample_val <= each_array[1]:
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
        loss_func = Alpha_Loss()
        running_loss_predict_tasks = 0
        num_iterations_predict_task = 0
        while not training_done:
            rand_timestep_within_sched = np.random.randint(cv_cutoff)
            set_of_twenty = self.start_of_each_set_twenty[rand_timestep_within_sched]
            truth = self.Y[set_of_twenty]

            which_schedule = self.find_which_schedule_this_belongs_to(set_of_twenty)
            load_in_embedding(self.model, self.embedding_list, which_schedule)
            phi_i_num = truth + set_of_twenty  # old method: set_of_twenty[0] + truth
            phi_i = self.X[phi_i_num]
            phi_i_numpy = np.asarray(phi_i)

            # variables to keep track of loss and number of tasks trained over

            # iterate over pairwise comparisons
            for counter in range(set_of_twenty, set_of_twenty + 20):
                if counter == phi_i_num:  # if counter == phi_i_num:
                    continue
                else:
                    phi_j = self.X[counter]
                    phi_j_numpy = np.asarray(phi_j)
                    feature_input = phi_i_numpy - phi_j_numpy
                    label = torch.ones((1, 1))

                    # label = add_noise_pairwise(label, self.noise_percentage)
                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)).cuda())
                        label = Variable(torch.Tensor(label).cuda())

                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)))
                        label = Variable(torch.Tensor(label.reshape(1, 1)))
                    output = self.model(feature_input)
                    if output > 1:
                        output = torch.floor(output)
                    self.opt.zero_grad()

                    loss = self.criterion(output, label)
                    print('loss is :', loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5) # clip any very high gradients

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
                    label = torch.zeros((1, 1))

                    if torch.cuda.is_available():
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)).cuda())
                        label = Variable(torch.Tensor(label).cuda())
                        label = label.reshape((1, 1))
                    else:
                        feature_input = Variable(torch.Tensor(feature_input.reshape(1, 12)))
                        label = Variable(torch.Tensor(label.reshape(1, 1)))
                    output = self.model(feature_input)


                    loss = self.criterion(output, label)
                    print('loss is :', loss)
                    # clip any very high gradients


                    # prepare optimizer, compute gradient, update params
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.opt.step()

                    running_loss_predict_tasks += loss.item()
                    num_iterations_predict_task += 1

            # add average loss to array
            print(list(self.model.parameters()))
            store_embedding_back(self.model, self.embedding_list, which_schedule)
            self.total_loss_array.append(running_loss_predict_tasks / num_iterations_predict_task)
            num_iterations_predict_task = 0
            running_loss_predict_tasks = 0

            self.total_iterations += 1

            print('total iterations is', self.total_iterations)
            if self.total_iterations > 25:
                print('total loss (average for each 40, averaged)', np.mean(self.total_loss_array[-20:]))

            if self.total_iterations > 0 and self.total_iterations % self.when_to_save == self.when_to_save - 1:
                # self.plot_nn()
                print(self.embedding_list)
                self.save_trained_nets()

            if self.total_iterations > 300000 and np.mean(self.total_loss_array[-100:]) - np.mean(
                    self.total_loss_array[-500:]) < self.covergence_epsilon:
                self.training_done = True

    def evaluate(self):
        pass

    def save_trained_nets(self):
        torch.save({'nn_state_dict': self.model.state_dict(), 'parameters': self.arguments}, '/home/ghost/PycharmProjects/bayesian_prolo/')


class Alpha_Loss(torch.nn.Module):
    def __init__(self):
        super(Alpha_Loss, self).__init__()

    def forward(self, P, Q, alpha):
        if alpha == 1:
            # diff = P - Q
            # totloss = torch.sum(torch.sum(torch.sum(diff)))
            # return totloss
            return (P * (P / Q).log()).sum()
        else:
            return 1/(alpha-1) * torch.log((P**alpha * Q **(1-alpha)).sum())
            # return 1 / (alpha - 1) * log((P * (Q / P) ** (1 - alpha)).sum())


def main():
    trainer = BDT_Train()
    trainer.train()


if __name__ == '__main__':
    main()
