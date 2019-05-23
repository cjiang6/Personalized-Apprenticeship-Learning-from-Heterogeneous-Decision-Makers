"""
naive approach for predicting action.
Will take in only the state and output the action.
"""

import os
import pickle
# sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')
import re
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from scheduling_env.argument_parser import Logger

sys.path.insert(0, '../')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)
np.random.seed(50)


class Naive_NN(nn.Module):
    def __init__(self):
        super(Naive_NN, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.fc1a = nn.Linear(128, 128)
        self.relu1a = nn.ReLU()
        self.fc1b = nn.Linear(128, 128)
        self.relu1b = nn.ReLU()
        self.fc1c = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(p=.2)
        self.relu1c = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 40)
        self.sig3 = nn.Sigmoid()

    def forward(self, x, dropout=True):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc1a(x)
        x = self.relu1a(x)
        x = self.fc1b(x)
        x = self.relu1b(x)
        x = self.fc1c(x)
        if dropout:
            x = self.dropout1(x)
        x = self.relu1c(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return self.sig3(x)  # 40 actions



class NaiveNNEvaluator():
    def __init__(self):
        self.arguments = Logger()
        self.loss_baseline_action = []
        self.sc2_max_training_iterations = self.arguments.sc2_max_training_iterations
        self.home_dir = "/home/ghost/PycharmProjects/scheduling_environment"

        # Directories
        self.all_data_train_dir = '/home/ghost/PycharmProjects/scheduling_environment/training_encoding_states_starcraft'
        self.all_data_test_dir = '/home/ghost/PycharmProjects/scheduling_environment/testing_encoding_states_starcraft'
        self.mmr = '/home/ghost/PycharmProjects/scheduling_environment/games_that_have_an_win_loss.pkl'
        self.list_of_games_mmr_train = pickle.load(open(self.mmr, "rb"))
        # This will produce list of games
        self.size_of_training_set = len(self.list_of_games_mmr_train)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nn_action = Naive_NN().to(device)


        self.optimizer = torch.optim.Adam(self.nn_action.parameters())


    def train(self):
        iteration = 0

        while iteration < 500:  # self.sc2_max_training_iterations:
            # remember only train protoss games
            # files in training directory
            files_in_training_directory = os.listdir(self.all_data_train_dir)
            # choose a batch of data
            batch = np.random.choice(files_in_training_directory)
            # open batch
            set_of_games = pickle.load(open(os.path.join(self.all_data_train_dir, batch), 'rb'))

            games = set_of_games['state_embeddings']  # state embedding dict
            actions = set_of_games['actions_at_each_frame']
            players = set_of_games['player_per_game']
            big_loss = set_of_games['big_loss']  # maybe games to skip
            crit = torch.nn.BCELoss()
            for l in range(50): # find a file within the batch
                # here we do bnn and nn
                file_with_low_loss_found = False
                filename = None

                # loop to find a filename with low loss (fairly inefficient)
                while not file_with_low_loss_found:
                    filename = np.random.choice(list(games.keys()))
                    if len(big_loss[filename]) < 50:
                        file_with_low_loss_found = True

                states_filename = games[filename]
                actions_filename = actions[filename]
                player_filename = players[filename]
                player_id = int(re.findall('\d+', batch)[0]) + list(games.keys()).index(filename)


                length_of_game = len(states_filename)
                # choose a random frame
                frame = np.random.randint(1, length_of_game - 1)  # random or sequence IDK
                X = states_filename[frame]
                action_taken_at_frame = actions_filename[frame]  # set of actions taken
                actions_taken_list = [i for i, e in enumerate(action_taken_at_frame) if e != 0]  # <-- only needed for pairwise
                # TODO: maybe add higher weighting for actions that are taken.



                if torch.cuda.is_available():
                    X = Variable(torch.Tensor(X).cuda())
                    Y = Variable(torch.Tensor(action_taken_at_frame).cuda())
                else:
                    X = Variable(torch.Tensor(X))
                    Y = Variable(torch.Tensor(action_taken_at_frame))

                prediction_baseline = self.nn_action.forward(X)


                # print(prediction_baseline)
                # print(prediction_bnn)

                loss = crit(prediction_baseline, Y)


                print("Current BASELINE loss for iteration  : ", iteration, "is", loss.item())

                self.optimizer.zero_grad()
                loss.backward()

                # Counter against SMART
                if actions_taken_list != [0]:
                    loss *= 5


                self.optimizer.step()

                self.loss_baseline_action.append(loss.item())


            iteration += 1
            if iteration % 50 == 49:
                # self.plot_networks()
                self.save()
    def save(self):
        torch.save({'nn_state_dict': self.nn_action.state_dict(),
                    'parameters': self.arguments},
                   '/home/ghost/PycharmProjects/bayesian_prolo/starcraft/nn_naive.tar')



    def test(self, load_in_model=True):
        if load_in_model == True:
            checkpoint = torch.load('/home/ghost/PycharmProjects/bayesian_prolo/starcraft/nn_naive.tar')
            self.nn_action.load_state_dict(checkpoint['nn_state_dict'])

        iteration = 0
        tot_test_games = 0
        max_frames = 0
        loss_lists_per_timestep_nn = []
        game_accuracies = []
        files_in_testing_directory = os.listdir(self.all_data_test_dir)
        for each_batch in files_in_testing_directory:
            print(each_batch)
            set_of_games = pickle.load(open(os.path.join(self.all_data_test_dir, each_batch), 'rb'))

            games = set_of_games['state_embeddings']  # state embedding dict
            actions = set_of_games['actions_at_each_frame']
            players = set_of_games['player_per_game']
            big_loss = set_of_games['big_loss']  # maybe games to skip
            loss = torch.nn.BCELoss()
            tot_test_games += len(games.keys())

            for filename in games.keys():
                if len(big_loss[filename]) > 50:
                    continue
                states_filename = games[filename]
                actions_filename = actions[filename]
                player_filename = players[filename]
                player_id = int(re.findall('\d+', each_batch)[0]) + list(games.keys()).index(filename)

                print("iteration ", iteration)
                frame = 0
                iteration+=1
                length_of_game = len(states_filename)
                if length_of_game > max_frames:
                    max_frames = length_of_game
                # choose a random frame
                while frame < length_of_game - 2:

                    frame += 1
                    X = states_filename[frame]
                    action_taken_at_frame = actions_filename[frame]  # set of actions taken
                    actions_taken_list = [i for i, e in enumerate(action_taken_at_frame) if e != 0]
                    if torch.cuda.is_available():
                        X = Variable(torch.Tensor(X).cuda())
                        Y = Variable(torch.Tensor(action_taken_at_frame).cuda())
                    else:
                        X = Variable(torch.Tensor(X))
                        Y = Variable(torch.Tensor(action_taken_at_frame))


                    prediction_baseline = self.nn_action.forward(X, dropout=False)

                    soft = nn.Softmax(dim=0)
                    prediction_baseline = soft(prediction_baseline)
                    baseline_loss = loss(prediction_baseline, Y)


                    print("Current BASELINE loss for iteration  : ", frame, "is", baseline_loss.item())

                    action_chosen_baseline = torch.argmax(prediction_baseline)

                    _, top_five_baseline = torch.topk(prediction_baseline, 5)

                    loss_lists_per_timestep_nn.append(baseline_loss.item())





                game_accuracies.append(np.mean(loss_lists_per_timestep_nn))
                loss_lists_per_timestep_nn = []
                # game has ended
                if len(game_accuracies) == 15:
                    self.print_and_store_accs(game_accuracies)
                    exit()
                # do end of schedule tings
            # a batch has ended
        # finished errthang

        print(tot_test_games)
        print(max_frames)
        self.print_and_store_accs(game_accuracies)

        # self.plot_with_errorbars()

    def print_and_store_accs(self, game_accs):
        print('Loss: {}'.format(np.mean(game_accs)))
        file = open('starcraft_learning_results.txt', 'a')
        file.write('NN_naive: mean: ' +
                   str(np.mean(game_accs)) +
                   ', std: ' + str(np.std(game_accs)) +
                   '\n')
        file.close()

def main():
    trainer = NaiveNNEvaluator()
    # trainer.train()
    trainer.test()

if __name__ == '__main__':
    main()







