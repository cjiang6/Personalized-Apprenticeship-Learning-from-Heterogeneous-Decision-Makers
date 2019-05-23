import sys
import torch
import sys
import torch.nn as nn
sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')
from starcraft.helper_funcs import *
from base_testing_environment.prolonet import ProLoNet

sys.path.insert(0, os.path.abspath('../..'))
from starcraft.argument_parser import Logger
# import matplotlib.pyplot as plt
import re
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)






class Trainer:
    def __init__(self):



        self.loss_array = []

        checkpoint = torch.load('/home/ghost/PycharmProjects/scheduling_environment/learn_action_embeddings.pkl')
        self.action_embedding_list = checkpoint['embedding_list']
        self.all_data_train_dir = '/home/ghost/PycharmProjects/scheduling_environment' + '/training_encoding_states_starcraft'


        self.mmr = '/home/ghost/PycharmProjects/scheduling_environment' + '/games_that_have_an_win_loss.pkl'
        self.list_of_games_mmr_train = pickle.load(open(self.mmr, "rb"))
        self.size_of_training_set = len(self.list_of_games_mmr_train)


        self.all_data_test_dir = '/home/ghost/PycharmProjects/scheduling_environment/testing_encoding_states_starcraft'


        self.gamma = .9
        self.criterion = torch.nn.BCELoss()
        self.not_converged = True
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding_list_Bayesian_neur_net = [torch.ones(12) * 1 / 2 for i in range(4000)]

        self.model = ProLoNet(input_dim=256+36,
                              weights=None,
                              comparators=None,
                              leaves=64,
                              output_dim=1,
                              bayesian_embedding_dim=12,
                              alpha=1.5,
                              use_gpu=True,
                              vectorized=True,
                              is_value=True)

        self.optimizer_main_net = torch.optim.Adam([{'params': list(self.model.parameters())[:-1]}, {'params': self.model.bayesian_embedding.parameters(), 'lr': .01}], lr=.001)
        self.embedding_optimizer_bnn = torch.optim.SGD(self.model.bayesian_embedding.parameters(), lr=.1)

    # noinspection PyArgumentList
    def train(self, num_steps_for_batch):
        sig = torch.nn.Sigmoid()
        iteration = 0
        while self.not_converged:

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

            for l in range(num_steps_for_batch):
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
                length_of_game = len(states_filename)

                # print("iteration ", iteration)

                # choose a random frame
                frame = np.random.randint(1, length_of_game)
                player_id = int(re.findall('\d+', batch)[0]) + list(games.keys()).index(filename)
                self.model.set_bayesian_embedding(self.embedding_list_Bayesian_neur_net[player_id])
                X = states_filename[frame]  # input - a
                actions_taken_at_frame = actions_filename[frame]  # set of actions taken
                actions_taken_list = [i for i, e in enumerate(actions_taken_at_frame) if e != 0]
                set_of_non_actions = self.compute_set_of_non_actions(actions_taken_list)

                # True
                Y = torch.ones((1, 1))  # label

                if torch.cuda.is_available():
                    X = Variable(torch.Tensor(X).cuda())
                    Y = Variable(torch.Tensor(Y).cuda())
                else:
                    X = Variable(torch.Tensor(X))
                    Y =  Variable(torch.Tensor(Y))


                running_loss = []
                for a in actions_taken_list:
                    action_embedding_a = self.action_embedding_list[int(a)]
                    for i, non_action in enumerate(set_of_non_actions):
                        action_embeddimg_a_prime = self.action_embedding_list[non_action]
                        subtracted_input = action_embedding_a.cuda() - action_embeddimg_a_prime.cuda()
                        prediction = self.model(torch.cat([X, subtracted_input.reshape(36)], dim=0).reshape(1,292))
                        prediction = sig(prediction)
                        prediction_loss = self.criterion(prediction, Y)
                        self.optimizer_main_net.zero_grad()
                        # if int(a) != 0:
                        #     prediction_loss *= 5
                        prediction_loss.backward()
                        self.optimizer_main_net.step()
                        running_loss.append(prediction_loss.item())

                # False
                Y = torch.zeros((1, 1))  # label
                if torch.cuda.is_available():
                    Y = Variable(torch.Tensor(Y).cuda())
                else:
                    Y =  Variable(torch.Tensor(Y))

                for a in actions_taken_list:
                    action_embedding_a = self.action_embedding_list[int(a)]
                    for i, non_action in enumerate(set_of_non_actions):
                        action_embeddimg_a_prime = self.action_embedding_list[non_action]
                        subtracted_input =  action_embeddimg_a_prime.cuda()- action_embedding_a.cuda()
                        prediction = self.model(torch.cat([X, subtracted_input.reshape(36)], dim=0).reshape(1,292))
                        prediction = sig(prediction)
                        # print(prediction)
                        prediction_loss = self.criterion(prediction, Y)
                        self.optimizer_main_net.zero_grad()
                        prediction_loss.backward()
                        self.optimizer_main_net.step()
                        running_loss.append(prediction_loss.item())
                if iteration > 55 and iteration % 50 == 49:
                    print('Mean loss for iteration: ', iteration, ' is ', np.mean(running_loss[-50:]))
                self.loss_array.append(np.mean(running_loss))

                self.embedding_list_Bayesian_neur_net[player_id] = torch.Tensor(self.model.get_bayesian_embedding().detach().cpu().numpy())  # very ugly

                iteration += 1
                if iteration % 1000 == 999:
                    # self.plot_networks()
                    self.save()

                if iteration > 6000 and np.mean(self.loss_array[-500:]) - np.mean(self.loss_array[-1000:]) < .001:
                    self.not_converged = False

    # noinspection PyArgumentList
    def test(self, load_in_model=False):
        iteration = 0
        tot_test_games = 0
        max_frames = 0
        loss_lists_per_timestep_nn = []
        sig = torch.nn.Sigmoid()
        game_accuracies = []
        embedding_list = [torch.ones(12) * 1 / 2 for i in range(4000)]
        if load_in_model == True:
            checkpoint = torch.load('/home/ghost/PycharmProjects/bayesian_prolo/starcraft/pairwise_sc2_PDDT.pkl')
            self.model.load_state_dict(checkpoint['state_dict'])
        files_in_testing_directory = os.listdir(self.all_data_test_dir)
        for each_batch in files_in_testing_directory:
            print(each_batch)
            set_of_games = pickle.load(open(os.path.join(self.all_data_test_dir, each_batch), 'rb'))


            games = set_of_games['state_embeddings']  # state embedding dict
            actions = set_of_games['actions_at_each_frame']
            players = set_of_games['player_per_game']
            big_loss = set_of_games['big_loss']  # maybe games to skip
            tot_test_games += len(games.keys())

            for game_num, filename in enumerate(games.keys()):
                if len(big_loss[filename]) > 50:
                    continue
                states_filename = games[filename]
                actions_filename = actions[filename]
                player_filename = players[filename]
                player_id = int(re.findall('\d+', each_batch)[0]) + list(games.keys()).index(filename)
                self.model.set_bayesian_embedding(embedding_list[player_id])
                print("iteration ", iteration)
                frame = 0
                iteration += 1
                length_of_game = len(states_filename)

                # choose a random frame
                while frame < length_of_game - 2:
                    self.probability_matrix_nn = np.zeros((40, 40))
                    print('reached frame ', frame)

                    # choose a random frame

                    X = states_filename[frame]  # input - a
                    actions_taken_at_frame = actions_filename[frame]  # set of actions taken
                    actions_taken_list = [i for i, e in enumerate(actions_taken_at_frame) if e != 0]
                    set_of_non_actions = self.compute_set_of_non_actions(actions_taken_list)

                    if torch.cuda.is_available():
                        X = Variable(torch.Tensor(X).cuda())

                    else:
                        X = Variable(torch.Tensor(X))

                    for i in range(40):
                        action_embedding_a = self.action_embedding_list[int(i)]
                        for j in range(40):
                            if i == j:
                                continue
                            else:
                                action_embedding_a_prime = self.action_embedding_list[int(j)]

                                subtracted_input = action_embedding_a.cuda() - action_embedding_a_prime.cuda()
                                prediction = self.model.forward(torch.cat([X, subtracted_input.reshape(36)], dim=0).reshape(1,292))
                                prediction = sig(prediction)
                                # add all these to matrixes

                                self.probability_matrix_nn[i][j] = prediction.item()


                    column_vec_nn = np.sum(self.probability_matrix_nn, axis=1)

                    loss = torch.nn.BCELoss()
                    soft = nn.Softmax(dim=0)
                    column_vec_nn = soft(torch.Tensor(column_vec_nn))
                    loss_nn = loss(column_vec_nn, Variable(torch.Tensor(actions_taken_at_frame)))

                    # loss_nn = np.linalg.norm(column_vec_nn - actions_taken_at_frame)


                    loss_lists_per_timestep_nn.append(loss_nn.item())
                    # True
                    Y = torch.ones((1, 1))  # label

                    if torch.cuda.is_available():
                        Y = Variable(torch.Tensor(Y).cuda())
                    else:
                        Y = Variable(torch.Tensor(Y))


                    for a in actions_taken_list:
                        action_embedding_a = self.action_embedding_list[int(a)]
                        for i, non_action in enumerate(set_of_non_actions):
                            action_embeddimg_a_prime = self.action_embedding_list[non_action]
                            subtracted_input = action_embedding_a.cuda() - action_embeddimg_a_prime.cuda()
                            prediction = self.model(torch.cat([X, subtracted_input.reshape(36)], dim=0).reshape(1, 292))
                            prediction = sig(prediction)
                            prediction_loss = self.criterion(prediction, Y)
                            self.embedding_optimizer_bnn.zero_grad()
                            prediction_loss.backward()
                            self.embedding_optimizer_bnn.step()


                    # False
                    Y = torch.zeros((1, 1))  # label
                    if torch.cuda.is_available():
                        Y = Variable(torch.Tensor(Y).cuda())
                    else:
                        Y = Variable(torch.Tensor(Y))

                    for a in actions_taken_list:
                        action_embedding_a = self.action_embedding_list[int(a)]
                        for i, non_action in enumerate(set_of_non_actions):
                            action_embeddimg_a_prime = self.action_embedding_list[non_action]
                            subtracted_input = action_embeddimg_a_prime.cuda() - action_embedding_a.cuda()
                            prediction = self.model(torch.cat([X, subtracted_input.reshape(36)], dim=0).reshape(1, 292))
                            prediction = sig(prediction)
                            prediction_loss = self.criterion(prediction, Y)
                            self.embedding_optimizer_bnn.zero_grad()
                            prediction_loss.backward()
                            self.embedding_optimizer_bnn.step()




                    frame += 1
                game_accuracies.append(np.mean(loss_lists_per_timestep_nn))
                loss_lists_per_timestep_nn = []
                self.print_and_store_accs(game_accuracies)
                if len(game_accuracies) == 5:
                    self.save_embeddings(embedding_list)
                if len(game_accuracies) == 15:
                    self.print_and_store_accs(game_accuracies)
                    self.save_embeddings()
                    exit()




                # game has ended


                # do end of schedule tings

            # a batch has ended
        # finished errthang

        print(tot_test_games)
        print(max_frames)


        # self.plot_with_errorbars()

    def print_and_store_accs(self, game_accs):
        print('Loss: {}'.format(np.mean(game_accs)))
        file = open('starcraft_learning_results.txt', 'a')
        file.write('PDDT_pairwise: mean: ' +
                   str(np.mean(game_accs)) +
                   ', std: ' + str(np.std(game_accs)) +
                   '\n')
        file.close()



    def save(self):
        torch.save({'state_dict': self.model.state_dict()},
                   '/home/ghost/PycharmProjects/bayesian_prolo/starcraft/pairwise_sc2_PDDT.pkl')

    def save_embeddings(self, embeddings):
        torch.save({'state_dict': self.model.state_dict(), 'embeddings': embeddings},
                   '/home/ghost/PycharmProjects/bayesian_prolo/starcraft/pairwise_sc2_PDDT_embeddings.pkl')

    def compute_set_of_non_actions(self, actions):
        set_of_non_actions = list(range(40))
        for each_action in actions:
            set_of_non_actions.remove(each_action)
        return set_of_non_actions


def main():

    trainer = Trainer()
    trainer.train(50)
    trainer.save()
    trainer.test()

if __name__ == '__main__':
    main()
