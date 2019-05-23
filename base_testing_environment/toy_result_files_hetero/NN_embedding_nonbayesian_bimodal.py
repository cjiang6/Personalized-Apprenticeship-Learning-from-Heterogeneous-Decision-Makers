"""
NN w/ Bayes rule updating for embedding implementation evaluating train and test performance on a homogeneous dataset
created on May 17, 2019 by Ghost
"""
from base_testing_environment.toy_result_files_hetero.generate_environment import create_simple_classification_dataset
from base_testing_environment.utils.accuracy_measures import compute_specificity, compute_sensitivity
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)  # ensures repeatability
np.random.seed(50)




class Classifier_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Classifier_MLP, self).__init__()
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
    #     self.h1 = nn.Linear(in_dim + 2, hidden_dim)
    #     self.h2 = nn.Linear(hidden_dim, hidden_dim)
    #     self.h3 = nn.Linear(hidden_dim, hidden_dim)
    #     self.out = nn.Linear(hidden_dim, out_dim)
    #     self.out_dim = out_dim
    #     self.bayesian_embedding = torch.Tensor([1, 0])
    #
    # def forward(self, x):
    #     x = torch.cat([x, self.bayesian_embedding.reshape(2)], dim=0)
    #     x = F.relu(self.h1(x))
    #     x = F.relu(self.h2(x))
    #     x = F.relu(self.h3(x))
    #     x = F.log_softmax(self.out(x))
    #     return x

    def set_bayesian_embedding(self, embedding_for_sched):
        """
        sets embedding inside BNN
        :param embedding_for_sched:
        :return:
        """
        self.bayesian_embedding = embedding_for_sched


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

def train():
    # Training set generation
    num_schedules = 50
    x_data, y = create_simple_classification_dataset(num_schedules, train=True)

    x = []
    for each_ele in x_data:
        x.append(each_ele[2:])

    x = torch.Tensor(x).reshape(-1, 2)
    y = torch.Tensor(y).reshape((-1, 1))

    print('Toy problem generated, and data cleaned')

    input_size = 2  # Just the x and z dimension
    hidden_size = 10  # The number of nodes at the hidden layer
    num_classes = 2  # The number of output classes. In this case, from 0 to 1
    learning_rate = 1e-3  # The speed of convergence

    MLP = Classifier_MLP(in_dim=input_size, hidden_dim=hidden_size, out_dim=num_classes)
    distributions = [np.ones(2) * 1 / 2 for _ in range(num_schedules)]
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(MLP.parameters(), lr=learning_rate)
    schedule_starts = np.linspace(0, int(num_schedules * 20 - 20), num=num_schedules)
    epochs = 150


    for epoch in range(epochs):  # loop over the dataset multiple times
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
                MLP.set_bayesian_embedding(each_i)
                tally = 1
                avg_loss_over_schedule = 0

                for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
                    x_t = x[each_t]
                    output = MLP.forward(x_t)
                    label = y[each_t]
                    label = torch.Tensor([label]).reshape(1)
                    label = label.long()

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

    print('Finished Training')
    return MLP

# REAL TEST
def test(MLP):
    x_data_test, y_test, percent_of_zeros = create_simple_classification_dataset(50, get_percent_of_zeros=True)
    schedule_starts = np.linspace(0, int(50 * 20 - 20), num=50)
    x_test = []

    for each_ele in x_data_test:
        x_test.append(each_ele[2:])

    x_test = torch.Tensor(x_test).reshape(-1, 2)
    y_test = torch.Tensor(y_test).reshape((-1, 1))

    test_losses, test_accs = [], []
    per_schedule_test_losses, per_schedule_test_accs = [], []
    preds, actual = [[] for _ in range(50)], [[] for _ in range(50)]
    test_distributions = [np.ones(2) * 1 / 2 for _ in range(50)]
    total_acc = []
    for i in range(50):
        chosen_schedule_start = int(schedule_starts[i])
        schedule_num = int(chosen_schedule_start / 20)
        embedding_given_dis, count = get_embedding_given_dist(test_distributions[schedule_num])
        prod = [.5, .5]
        acc = 0
        MLP.set_bayesian_embedding(embedding_given_dis)

        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            # at each timestep you what to resample the embedding

            x_t = x_test[each_t]
            output = MLP.forward(x_t)

            label = y_test[each_t]
            label = torch.Tensor([label]).reshape(1)
            label = label.long()
            print('output is ', torch.argmax(output).item(), ' label is ', label.item())
            preds[i].append(torch.argmax(output).item())
            actual[i].append(label.item())
            if torch.argmax(output).item() == label.item():
                acc += 1
            tally = output[int(label.item())].item()
            second_tally = output[int(not label.item())].item()
            prod[count] = tally * test_distributions[i][count]
            prod[int(not count)] *= second_tally * test_distributions[i][int(not count)]

            normalization_factor = sum(prod)
            prod = [k / normalization_factor for k in prod]

            test_distributions[schedule_num][0] = prod[0]
            test_distributions[schedule_num][1] = prod[1]
            normalization_factor_for_dist = sum(test_distributions[schedule_num])
            test_distributions[schedule_num] /= normalization_factor_for_dist  # [i/normalization_factor_for_dist for i in distributions[schedule_num]]
            print('distribution at time ', each_t, ' is', test_distributions[schedule_num])
            if each_t % 20 < 5:
                embedding_given_dis, count = get_embedding_given_dist(test_distributions[schedule_num])
            else:
                embedding_given_dis = get_most_likely_embedding_given_dist(test_distributions[schedule_num])
            MLP.set_bayesian_embedding(embedding_given_dis)

        per_schedule_test_accs.append(acc / 20)
    # print('Loss: {}, Accuracy: {}'.format(0, np.mean(per_schedule_test_accs)))

    sensitivity, specificity = compute_sensitivity(preds, actual), compute_specificity(preds, actual)
    print('per sched accuracy: ', np.mean(per_schedule_test_accs))
    print('mean sensitivity: ', sensitivity, ', mean specificity: ', specificity)
    file = open('heterogeneous_toy_env_results.txt', 'a')
    file.write('NN w/ bimodal embedding: mean: ' +
               str(np.mean(per_schedule_test_accs)) +
               ', std: ' + str(np.std(per_schedule_test_accs)) +
               ', sensitivity: ' + str(sensitivity) + ', specificity: ' + str(specificity) +
               ', Distribution of Class: 0: ' + str(percent_of_zeros) + ', 1: ' + str(1 - percent_of_zeros) +
               '\n')
    file.close()



def main():
    bnn = train()
    test(bnn)

if __name__ == '__main__':
    main()
