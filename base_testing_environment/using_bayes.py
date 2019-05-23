"""
Testing new method
"""
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from sklearn.tree import DecisionTreeClassifier
from base_testing_environment.toy_result_files_hetero.generate_environment import create_simple_classification_dataset

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)
np.random.seed(50)


class NeuronBNN(nn.Module):
    def __init__(self):
        super(NeuronBNN, self).__init__()

        self.fc1 = nn.Linear(4, 1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4, 1)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(3, 2)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(2, 2)
        self.soft = nn.Softmax()
        self.bayesian_embedding = torch.Tensor([1, 0])  # This will always be 1 or 0
        # bayes_embed.requires_grad = True
        # self.bayesian_embedding = nn.Parameter(bayes_embed)

    def forward(self, x):
        """

        :param x: lets specify x is made of lambda, z, x
        :return:
        """
        x = torch.cat([x, self.bayesian_embedding.reshape(2)], dim=0)
        x1 = self.fc1(x)
        x1 = self.relu1(x1)
        x2 = self.fc2(x)
        x2 = self.relu2(x2)
        x3 = self.fc3(x)
        x3 = self.relu3(x3)
        x = self.fc4(torch.cat([x1, x2, x3]))
        x = self.relu(x)
        x = self.fc5(x)
        x = self.soft(x)
        return x

    def set_bayesian_embedding(self, embedding_for_sched):
        """
        sets embedding inside BNN
        :param embedding_for_sched:
        :return:
        """
        self.bayesian_embedding = embedding_for_sched

    def set_weights_and_bias(self):
        """
        sets weights and biases in order to give importance to embedding
        :return:
        """

        self.fc1.weight.data[0][2].fill_(4)
        self.fc1.weight.data[0][3].fill_(-10)

        self.fc2.weight.data[0][2].fill_(-2)
        self.fc2.weight.data[0][3].fill_(10)

        self.fc3.weight.data[0][2].fill_(2)
        self.fc3.weight.data[0][2].fill_(-6)




def get_embedding_given_dist(dist):
    choice = np.random.choice([0, 1], p=[dist[0], dist[1]])
    if choice == 0:
        return torch.Tensor([1, 0]), 0
    else:
        return torch.Tensor([0, 1]), 1


def get_most_likely_embedding_given_dist(dist):
    choice = np.argmax(dist)
    if choice == 0:
        return torch.Tensor([1, 0])
    else:
        return torch.Tensor([0, 1])


# noinspection PyArgumentList
def train(model, num_schedules, data, labels):
    """
    Pr[omega_i | game_g] = Pr[omega = omega_i] * \Prod_{data points j in game g} Pr[y_j | omega_i , x_j]

    Pr[assignment_g^{t+1} = i] = Pr[assignment_g^t = i] * Pr[omega_i | game g]
    :param model:
    :param num_schedules:
    :param data:
    :param labels:
    :return:
    """
    distributions = [np.ones(2) * 1 / 2 for _ in range(num_schedules)]
    epochs = 200
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=.001)
    schedule_starts = np.linspace(0, int(num_schedules*20-20), num=num_schedules)
    for epoch in range(epochs):
        for _ in range(num_schedules):
            # choose a schedule
            chosen_schedule_start = int(np.random.choice(schedule_starts))
            schedule_num = int(chosen_schedule_start / 20)
            # embedding_given_dis = get_embedding_given_dist(distributions[schedule_num])
            prod = [0, 0]
            # for each embedding
            opt.zero_grad()
            test_models = [0, 0]
            losses = [0, 0]
            for count, each_i in enumerate([torch.Tensor([1, 0]), torch.Tensor([0, 1])]):
                # model_copy = NeuronBNN()
                # model_copy.load_state_dict(model.state_dict())
                # test_models[count] = model_copy
                model.set_bayesian_embedding(each_i)
                tally = 1
                avg_loss_over_schedule = 0

                for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
                    x = data[each_t][2:]
                    x = torch.Tensor([x]).reshape((2))
                    output = model.forward(x)
                    label = labels[each_t]
                    label = torch.Tensor([label]).reshape(1)
                    label = Variable(label).long()

                    loss = criterion(output.reshape(1, 2), label)
                    # loss.backward()
                    losses[count] += loss
                    avg_loss_over_schedule += loss.item()

                    # print('output is ', torch.argmax(output).item(), ' label is ', label.item())
                    tally *= output[int(label.item())].item()
                prod[count] = tally * distributions[schedule_num][count]
                if max(distributions[schedule_num]) == distributions[schedule_num][count]:  # if you choose correctly
                    print("avg loss over schedule is: ", avg_loss_over_schedule / 20)

            normalization_factor = sum(prod)
            prod = [i / normalization_factor for i in prod]

            # update each
            distributions[schedule_num][0] = prod[0]
            distributions[schedule_num][1] = prod[1]

            normalization_factor_for_dist = sum(distributions[schedule_num])
            distributions[schedule_num] /= normalization_factor_for_dist  # [i/normalization_factor_for_dist for i in distributions[schedule_num]]
            coin_flip = np.random.choice([0, 1], p=[distributions[schedule_num][0], distributions[schedule_num][1]])
            chosen_loss = losses[int(coin_flip)]
            chosen_loss.backward()
            # opt = torch.optim.SGD(test_models[coin_flip].parameters(), lr=.001)
            opt.step()
            # model.load_state_dict(test_models[coin_flip].state_dict())

    print(distributions)
    print('finite')

    #
    # for epoch in range(epochs):
    #     for _ in range(num_schedules):
    #         # choose a schedule
    #         chosen_schedule_start = int(np.random.choice(schedule_starts))
    #         schedule_num = int(chosen_schedule_start / 20)
    #         embedding_given_dis, count = get_embedding_given_dist(distributions[schedule_num])
    #
    #         model.set_bayesian_embedding(embedding_given_dis)
    #
    #         avg_loss_over_schedule = 0
    #         for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
    #             x = data[each_t][2:]
    #             x = torch.Tensor([x]).reshape((2))
    #             output = model.forward(x)
    #             label = labels[each_t]
    #             label = torch.Tensor([label]).reshape(1)
    #             label = Variable(label).long()
    #             opt.zero_grad()
    #             loss = criterion(output.reshape(1, 2), label)
    #             if torch.argmax(output).item() == 0:
    #                 loss *= 100
    #             avg_loss_over_schedule += loss.item()
    #             loss.backward()
    #             opt.step()
    #
    #         print("avg loss over schedule is: ", avg_loss_over_schedule / 20)
    #
    #
    # print('finite')


def test(model, num_schedules, data, labels):
    """
    Pr[omega_i | game_g] = Pr[omega = omega_i] * \Prod_{data points j in game g} Pr[y_j | omega_i , x_j]
    Pr[assignment_g^{t+1} = i] = Pr[assignment_g^t = i] * Pr[omega_i | game g]
    :param model:
    :param num_schedules:
    :param data:
    :param labels:
    :return:
    """
    distributions = [np.ones(2) * 1 / 2 for _ in range(num_schedules)]
    schedule_starts = np.linspace(0, int(num_schedules*20-20), num=num_schedules)
    total_acc = []
    for i in range(num_schedules):
        # choose a schedule
        chosen_schedule_start = int(schedule_starts[i])
        schedule_num = int(chosen_schedule_start / 20)
        embedding_given_dis, count = get_embedding_given_dist(distributions[schedule_num])
        prod = [.5, .5]
        model.set_bayesian_embedding(embedding_given_dis)
        tally = 1
        second_tally = 1
        acc = 0
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            # at each timestep you what to resample the embedding

            x = data[each_t][2:]

            x = torch.Tensor([x]).reshape((2))
            output = model.forward(x)

            label = labels[each_t]
            label = torch.Tensor([label]).reshape(1)
            label = Variable(label).long()
            print('output is ', torch.argmax(output).item(), ' label is ', label.item())
            if torch.argmax(output).item() == label.item():
                acc += 1
            tally = output[int(label.item())].item()
            second_tally = output[int(not label.item())].item()
            prod[count] = tally * distributions[i][count]
            prod[int(not count)] *= second_tally * distributions[i][int(not count)]

            normalization_factor = sum(prod)
            prod = [k / normalization_factor for k in prod]

            distributions[schedule_num][0] = prod[0]
            distributions[schedule_num][1] = prod[1]
            normalization_factor_for_dist = sum(distributions[schedule_num])
            distributions[schedule_num] /= normalization_factor_for_dist  # [i/normalization_factor_for_dist for i in distributions[schedule_num]]
            print('distribution at time ', each_t, ' is', distributions[schedule_num])
            if each_t % 20 < 5:
                embedding_given_dis, count = get_embedding_given_dist(distributions[schedule_num])
            else:
                embedding_given_dis = get_most_likely_embedding_given_dist(distributions[schedule_num])
            model.set_bayesian_embedding(embedding_given_dis)
        total_acc.append(acc / 20)
    print('mean is ', np.mean(total_acc))
    print('finite')


def DTtrain(num_schedules, data, labels):
    """
    Pr[omega_i | game_g] = Pr[omega = omega_i] * \Prod_{data points j in game g} Pr[y_j | omega_i , x_j]

    Pr[assignment_g^{t+1} = i] = Pr[assignment_g^t = i] * Pr[omega_i | game g]
    :param num_schedules:
    :param data:
    :param labels:
    :return:
    """
    distributions = [np.ones(2) * 1 / 2 for _ in range(num_schedules)]
    epochs = 1000
    schedule_starts = np.linspace(0, 980, num=num_schedules)
    clf = None
    for epoch in range(epochs):
        # data augmentation
        augmented_data = []
        answers = []
        which_embedding_was_chosen = []
        for j in range(num_schedules):
            # choose a schedule
            chosen_schedule_start = int(schedule_starts[j])
            schedule_num = int(chosen_schedule_start / 20)
            embedding_given_dis, count = get_embedding_given_dist(distributions[schedule_num])

            for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
                x = data[each_t][2:]

                x = torch.Tensor([x]).reshape((2))
                x = list(np.array(torch.cat([x, embedding_given_dis])))
                augmented_data.append(x)
                answers.append(labels[each_t][0])
                which_embedding_was_chosen.append(count)
        clf = DecisionTreeClassifier(max_depth=6)
        clf.fit(augmented_data, answers)

        for j in range(num_schedules):
            # choose a schedule
            chosen_schedule_start = int(schedule_starts[j])
            schedule_num = int(chosen_schedule_start / 20)
            prod = [0, 0]
            # for each embedding
            count = which_embedding_was_chosen[chosen_schedule_start]
            tally = [1, 1]
            for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
                data_for_timestep = augmented_data[each_t]
                label_for_timestep = answers[each_t]

                y_pred = clf.predict_proba(np.array(data_for_timestep).reshape(1,-1))

                print('output is ', y_pred[0], ' label is ', label_for_timestep)

                tally[count] *= y_pred[0][label_for_timestep]
                tally[int(not count)] *= y_pred[0][int(not label_for_timestep)]

            prod[count] = tally[count] * distributions[schedule_num][count]
            prod[int(not count)] = tally[int(not count)] * distributions[schedule_num][int(not count)]

            normalization_factor = sum(prod)
            prod = [i / normalization_factor for i in prod]

            # update each
            distributions[schedule_num][0] = prod[0]
            distributions[schedule_num][1] = prod[1]

            normalization_factor_for_dist = sum(distributions[schedule_num])
            distributions[schedule_num] /= normalization_factor_for_dist  # [i/normalization_factor_for_dist for i in distributions[schedule_num]]


    print(distributions)
    print('finite')
    return clf

    #
    # for epoch in range(epochs):
    #     for _ in range(num_schedules):
    #         # choose a schedule
    #         chosen_schedule_start = int(np.random.choice(schedule_starts))
    #         schedule_num = int(chosen_schedule_start / 20)
    #         embedding_given_dis, count = get_embedding_given_dist(distributions[schedule_num])
    #
    #         model.set_bayesian_embedding(embedding_given_dis)
    #
    #         avg_loss_over_schedule = 0
    #         for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
    #             x = data[each_t][2:]
    #             x = torch.Tensor([x]).reshape((2))
    #             output = model.forward(x)
    #             label = labels[each_t]
    #             label = torch.Tensor([label]).reshape(1)
    #             label = Variable(label).long()
    #             opt.zero_grad()
    #             loss = criterion(output.reshape(1, 2), label)
    #             if torch.argmax(output).item() == 0:
    #                 loss *= 100
    #             avg_loss_over_schedule += loss.item()
    #             loss.backward()
    #             opt.step()
    #
    #         print("avg loss over schedule is: ", avg_loss_over_schedule / 20)
    #
    #
    # print('finite')


def DTtest(tree, num_schedules, data, labels):
    """
    Pr[omega_i | game_g] = Pr[omega = omega_i] * \Prod_{data points j in game g} Pr[y_j | omega_i , x_j]
    Pr[assignment_g^{t+1} = i] = Pr[assignment_g^t = i] * Pr[omega_i | game g]
    :param tree:
    :param num_schedules:
    :param data:
    :param labels:
    :return:
    """
    distributions = [np.ones(2) * 1 / 2 for _ in range(num_schedules)]
    schedule_starts = np.linspace(0, 980, num=num_schedules)
    total_acc = []
    for i in range(num_schedules):
        # choose a schedule
        chosen_schedule_start = int(schedule_starts[i])
        schedule_num = int(chosen_schedule_start / 20)
        embedding_given_dis, count = get_embedding_given_dist(distributions[schedule_num])
        prod = [.5, .5]

        acc = 0
        tally = [1,1]
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            # at each timestep you what to resample the embedding

            x = data[each_t][2:]

            x = torch.Tensor([x]).reshape((2))
            x = list(np.array(torch.cat([x, embedding_given_dis])))
            y_pred = tree.predict_proba(np.array(x).reshape(1,-1))

            label = labels[each_t]

            print('output is ', y_pred[0], ' label is ', label)
            if np.argmax(y_pred[0]) == label:
                acc += 1
            tally[count] *= y_pred[0][int(label[0])]
            tally[int(not count)] *= y_pred[0][int(not label[0])]
            prod[count] = tally[count] * distributions[schedule_num][count]
            prod[int(not count)] = tally[int(not count)] * distributions[schedule_num][int(not count)]

            normalization_factor = sum(prod)
            prod = [k / normalization_factor for k in prod]

            distributions[schedule_num][0] = prod[0]
            distributions[schedule_num][1] = prod[1]
            normalization_factor_for_dist = sum(distributions[schedule_num])
            distributions[schedule_num] /= normalization_factor_for_dist  # [i/normalization_factor_for_dist for i in distributions[schedule_num]]

            print('distribution at time ', each_t, ' is', distributions[schedule_num])
            if each_t % 20 < 5:
                embedding_given_dis, count = get_embedding_given_dist(distributions[schedule_num])
            else:
                embedding_given_dis = get_most_likely_embedding_given_dist(distributions[schedule_num])
                count = np.argmax(distributions[schedule_num])

        total_acc.append(acc / 20)
    print('mean is ', np.mean(total_acc))
    print('finite')


def main():
    num_schedules = 50
    data, labels = create_simple_classification_dataset(num_schedules, train=True)
    bnn = NeuronBNN()
    # bnn.set_weights_and_bias()
    train(bnn, num_schedules, data, labels)
    test_data, test_labels = create_simple_classification_dataset(50)
    test(bnn, 50, test_data, test_labels)


def DTmain():
    num_schedules = 50
    data, labels = create_simple_classification_dataset(num_schedules, train=True)
    # bnn.set_weights_and_bias()
    tree=DTtrain(num_schedules, data, labels)
    test_data, test_labels = create_simple_classification_dataset(50)
    DTtest(tree, 50, test_data, test_labels)


if __name__ == '__main__':
    DTmain()
