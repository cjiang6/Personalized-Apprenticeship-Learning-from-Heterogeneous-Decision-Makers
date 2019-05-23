"""
The other thing is that this is taking a while (hours) for each test.
Could you come up with a trivial data set where the input is x and omega the output is either y,
so we have y = f(x,omega), which is nothing new. However, then you literally have x be only a single dimension,
y be a single dimension, and omega be a single dimension, and you literally have y = f(x,omega) = x*omega.
You have x \in [0,1] and omega \in {-1,1}. Then the answer should be either x or -x depending on omega.
"""
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from utils.global_utils import save_pickle, load_in_embedding_bnn, store_embedding_back_bnn

np.random.seed(50)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)


# noinspection PyTypeChecker
class EmbeddingModule(nn.Module):
    """
    embedding class (allows us to access parameters directly)
    """

    def __init__(self):
        super(EmbeddingModule, self).__init__()
        self.embedding = nn.Parameter(torch.ones(1, 1) * .5)

    def forward(self):
        """
        doesn't do anything
        :return:
        """
        return


# noinspection PyTypeChecker
class basic_bnn(nn.Module):
    def __init__(self):
        super(basic_bnn, self).__init__()
        # self.fc1 = nn.Linear(2, 1)
        self.EmbeddingList = nn.ModuleList(EmbeddingModule() for _ in range(1))

    def forward(self, x):
        w = self.EmbeddingList[0].embedding
        # x = torch.cat([x, w], dim=1)
        # x = self.fc1(x)
        return x * w


# noinspection PyTypeChecker
class basic_bnn_net(nn.Module):
    def __init__(self):
        super(basic_bnn_net, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(2, 2)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(2, 1)
        self.EmbeddingList = nn.ModuleList(EmbeddingModule() for _ in range(1))

    def forward(self, x):
        w = self.EmbeddingList[0].embedding
        x = torch.cat([x, w], dim=1)
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        x = self.fc3(x)
        return x



class basic_2_diff_bnn(nn.Module):
    def __init__(self):
        super(basic_2_diff_bnn, self).__init__()
        # self.fc1 = nn.Linear(2, 1)
        self.EmbeddingList = nn.ModuleList(EmbeddingModule() for _ in range(1))

    def forward(self, x1, x2):
        w = self.EmbeddingList[0].embedding
        # x = torch.cat([x, w], dim=1)
        # x = self.fc1(x)
        return x1 * w + x2
        # NOTE: this is given the rule



class basic_2_diff_no_rule_bnn(nn.Module):
    def __init__(self):
        super(basic_2_diff_no_rule_bnn, self).__init__()
        # self.fc1 = nn.Linear(2, 1)
        self.EmbeddingList = nn.ModuleList(EmbeddingModule() for _ in range(1))

    def forward(self, x1, x2):
        w = self.EmbeddingList[0].embedding
        x = torch.cat([x1, w], dim=1)
        x = self.fc1(x)
        return x1 * w + x2
        # NOTE: this is given the rule


# noinspection PyTypeChecker
class DosNeuronBNN(nn.Module):
    def __init__(self):
        super(DosNeuronBNN, self).__init__()
        self.fc1 = nn.Linear(3, 1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2, 1)

    def forward(self, x):
        """

        :param x: lets specify x is made of lambda, z, x
        :return:
        """
        x1 = self.fc1(x)
        x1 = self.relu1(x1)
        x2 = self.fc2(x)
        x2 = self.relu2(x2)
        x3 = self.fc3(torch.cat([x1, x2]))
        return x3

    def set_weights_and_bias(self):
        self.fc1.weight.data[0][0].fill_(1000)
        self.fc1.weight.data[0][1].fill_(1)
        self.fc1.weight.data[0][2].fill_(1000)

        self.fc2.weight.data[0][0].fill_(-1000)
        self.fc2.weight.data[0][1].fill_(-1)
        self.fc2.weight.data[0][2].fill_(1000)

        self.fc1.bias.data.fill_(-2000)

        self.fc2.bias.data.fill_(-998)

        self.fc3.bias.data.fill_(0)
        self.fc3.weight.data[0][0].fill_(1)
        self.fc3.weight.data[0][1].fill_(1)


def func(x, omega):
    return x * omega


def get_omega():
    omega = np.random.choice([-1, 1])
    return omega


def get_x(n):
    x = np.random.uniform(0, 1, (1, n))
    return x


def create_dataset():
    dataset = []
    omegas = []
    num_segments = 100
    while len(dataset) < 1000:
        omega = get_omega()
        x = get_x(num_segments)
        output = omega * x

        dataset.extend(list(output[0]))
        for i in range(num_segments):
            omegas.append(omega)
    # save_pickle(file_location='/home/ghost/PycharmProjects/bayesian_prolo', file={'omegas': omegas, 'dataset': dataset}, special_string='super_simple_dataset')
    return omegas, dataset


def create_2_difficulty_dataset():
    dataset = []
    omegas = []
    num_segments = 100
    while len(dataset) < 1000:
        omega = get_omega()
        x1 = get_x(num_segments)
        x2 = get_x(num_segments)
        output = omega * x1 + x2

        dataset.extend(list(output[0]))
        for i in range(num_segments):
            omegas.append(omega)
    # save_pickle(file_location='/home/ghost/PycharmProjects/bayesian_prolo', file={'omegas': omegas, 'dataset': dataset}, special_string='super_simple_dataset')
    return omegas, dataset


def create_dual_neuron_dataset(n):
    # sample z from 0 to 1
    lst = [[] for i in range(n * 20)]  # each list is a timestep
    label = [[] for i in range(n * 20)]  # each list is a timestep
    for i in range(n):
        if i % 2 == 0:
            lam = 1
        else:
            lam = 0

        for count in range(20):
            z = np.random.uniform(0, 1)
            x = np.random.choice([0, 1])
            if lam == 1:
                y = z * x
            else:
                y = (2 - z) * x

            lst[i * 20 + count].extend([lam, z, x])
            label[i * 20 + count].append(y)

    return lst, label

def test_inital_given_weights_and_belief():
    bnn = DosNeuronBNN()
    bnn.set_weights_and_bias()
    # 5 examples
    x1 = torch.Tensor([1, .5, 1])  # expect .5
    x2 = torch.Tensor([0, .5, 1])  # expect 1.5
    x3 = torch.Tensor([1, .5, 0])  # expect 0
    x4 = torch.Tensor([0, .5, 0])  # expect 0
    x5 = torch.Tensor([1, .25, 1])  # expect .25
    x6 = torch.Tensor([0, 1, 1])  # expect 1
    x7 = torch.Tensor([1, 1, 0])  # expect 0

    for i in [x1, x2, x3, x4, x5, x6, x7]:
        print('sol is', bnn.forward(i.reshape((3))).item())

# Works fine, now let's try giving it lambda but removing the weight

def give_lambda_but_allow_weights_to_train():
    # we need to create a dataset first

    data, labels = create_dual_neuron_dataset(100)

    # reinitialize bnn
    bnn = DosNeuronBNN()
    bnn.__init__()

    params = list(bnn.parameters())
    opt = torch.optim.SGD(params, lr=.0001)

    loss = nn.L1Loss()  # can use L1 as well, shouldn't matter too much
    epochs = 1

    # even sets of twenty are lam = 1
    # odd sets of twenty are lam = 0
    even_lambdas = np.linspace(0,1960,num=50)
    for epoch in range(epochs):
        for j in range(5):
            # chose an even schedule
            even = int(np.random.choice(even_lambdas))
            # for each schedules
            for i in range(even, even + 20):
                # load in data
                x = data[i]
                print('lambda is : ', x[0])
                label = labels[i]
                x = torch.Tensor([x]).reshape((3))
                label = torch.Tensor([label]).reshape((1, 1))
                output = bnn.forward(x)
                opt.zero_grad()
                error = loss(output, label)
                print('loss is: ', error.item())
                error.backward()
                # weight step
                opt.step()

        for j in range(5):
            # chose an even schedule
            odd = int(np.random.choice(even_lambdas)) + 20
            for i in range(odd, odd+20):
                x = data[i]
                label = labels[i]
                x = torch.Tensor([x]).reshape((3))
                label = torch.Tensor([label]).reshape((1, 1))
                output = bnn.forward(x)
                opt.zero_grad()
                error = loss(output, label)
                print('loss is: ', error.item())
                error.backward()
                opt.step()


    test_data, test_labels = create_dual_neuron_dataset(20)
    print(bnn.state_dict())
    avg_loss = 0

    for i in range(20 * 20):
        x = test_data[i]
        label = test_labels[i]
        x = torch.Tensor([x]).reshape((3))
        label = torch.Tensor([label]).reshape((1, 1))
        output = bnn.forward(x)

        error = loss(output, label)
        print('output is ', output)
        print('label is ', label)
        print(error)
        avg_loss += error.item()
    avg_loss /= 400
    print('avg loss is', avg_loss)


# Now we make allow the embedding to be a feature.
# but still set the weights


# I'll redefine the network to make things easy

# noinspection PyTypeChecker
class NeuronBNN(nn.Module):
    def __init__(self):
        super(NeuronBNN, self).__init__()
        self.fc1 = nn.Linear(3, 1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2, 1)
        bayes_embed = torch.Tensor(.5)
        bayes_embed.requires_grad = True
        self.bayesian_embedding = nn.Parameter(bayes_embed)

    def forward(self, x):
        """

        :param x: lets specify x is made of lambda, z, x
        :return:
        """
        w = self.EmbeddingList[0].embedding
        x = torch.cat([x, w.reshape(1)], dim=0)
        x1 = self.fc1(x)
        x1 = self.relu1(x1)
        x2 = self.fc2(x)
        x2 = self.relu2(x2)
        x3 = self.fc3(torch.cat([x1, x2]))
        return x3

    def set_weights_and_bias(self):
        self.fc1.weight.data[0][0].fill_(1000)
        self.fc1.weight.data[0][1].fill_(1)
        self.fc1.weight.data[0][2].fill_(1000)

        self.fc2.weight.data[0][0].fill_(-1000)
        self.fc2.weight.data[0][1].fill_(-1)
        self.fc2.weight.data[0][2].fill_(1000)

        self.fc1.bias.data.fill_(-2000)

        self.fc2.bias.data.fill_(-998)

        self.fc3.bias.data.fill_(0)
        self.fc3.weight.data[0][0].fill_(1)
        self.fc3.weight.data[0][1].fill_(1)


def give_proper_weights_only_allow_embedding_to_tune():
    bnn = NeuronBNN()
    bnn.set_weights_and_bias()
    # we need to create a dataset first

    data, labels = create_dual_neuron_dataset(100)

    embedding_opt = torch.optim.SGD(bnn.EmbeddingList[0].parameters(), lr=.9)

    embedding_list = [torch.ones(1, 1) * 1/2 for _ in range(100)]
    loss = nn.L1Loss()  # can use L1 as well, shouldn't matter too much
    epochs = 1000
    # sorry for the copy paste
    # even sets of twenty are lam = 1
    # odd sets of twenty are lam = 0
    even_lambdas = np.linspace(0,1960,num=50)
    for epoch in range(epochs):
        for j in range(1):
            # chose an even schedule
            even = int(np.random.choice(even_lambdas))
            load_in_embedding_bnn(bnn, embedding_list, int(even /20))
            for i in range(even, even + 20):
                x = data[i][0:2]
                label = labels[i]
                x = torch.Tensor([x]).reshape((2))
                label = torch.Tensor([label]).reshape((1, 1))
                output = bnn.forward(x)

                # opt.zero_grad()
                embedding_opt.zero_grad()
                error = loss(output, label)
                error.backward()
                embedding_opt.step()
                # opt.step()

            embedding_list = store_embedding_back_bnn(bnn, embedding_list, int(even / 20))

        for j in range(1):
            # chose an even schedule
            odd = int(np.random.choice(even_lambdas)) + 20
            load_in_embedding_bnn(bnn, embedding_list, int(odd / 20))
            for i in range(odd, odd+20):
                x = data[i][0:2]
                label = labels[i]
                x = torch.Tensor([x]).reshape((2))
                label = torch.Tensor([label]).reshape((1, 1))
                output = bnn.forward(x)
                # opt.zero_grad()
                embedding_opt.zero_grad()
                error = loss(output, label)
                error.backward()
                embedding_opt.step()
                # opt.step()
            embedding_list = store_embedding_back_bnn(bnn, embedding_list, int(odd / 20))


    test_data, test_labels = create_dual_neuron_dataset(20)
    print(bnn.state_dict())
    avg_loss = 0
    test_embedding_list = [torch.ones(1, 1) * 1/2 for _ in range(20)]
    embedding_opt = torch.optim.SGD(bnn.EmbeddingList[0].parameters(), lr=.1)
    counter = 0
    for i in range(20 * 20):
        load_in_embedding_bnn(bnn, test_embedding_list, int(i / 20))
        x = test_data[i][0:2]
        label = test_labels[i]
        x = torch.Tensor([x]).reshape((2))
        label = torch.Tensor([label]).reshape((1, 1))
        output = bnn.forward(x)

        error = loss(output, label)
        print('output is ', output)
        print('label is ', label)
        print('error is ', error.item())
        avg_loss += error.item()
        if error.item() < .05:
            counter +=1
        if error.item() > .05:
            flag = False
            tracker = 0
            while not flag:
                embedding_opt.zero_grad()
                error.backward()
                embedding_opt.step()
                output = bnn.forward(x)
                error = loss(output, label)
                tracker += 1
                if tracker > 100:
                    flag = True
                if error.item() < .1:
                    flag = True
        test_embedding_list = store_embedding_back_bnn(bnn, test_embedding_list, int(i / 20))


    print(test_embedding_list)
    print(embedding_list)
    avg_loss /= 400
    print('avg loss is', avg_loss)
    print ('accuracy', counter/400)



give_proper_weights_only_allow_embedding_to_tune()




































def tune_both():

    bnn = NeuronBNN()
    # now we let network tune both
    bnn.__init__()
    data, labels = create_dual_neuron_dataset(100)

    params = list(bnn.parameters())
    del params[6]
    opt = torch.optim.SGD(params, lr=.0001)
    embedding_opt = torch.optim.SGD(bnn.EmbeddingList[0].parameters(), lr=.01)

    embedding_list = [torch.ones(1, 1) * 1/2 for _ in range(100)]
    loss = nn.L1Loss()  # can use L1 as well, shouldn't matter too much
    epochs = 1000
    # sorry for the copy paste
    # even sets of twenty are lam = 1
    # odd sets of twenty are lam = 0
    even_lambdas = np.linspace(0,1960,num=50)
    for epoch in range(epochs):
        for j in range(5):
            # chose an even schedule
            even = int(np.random.choice(even_lambdas))
            load_in_embedding_bnn(bnn, embedding_list, int(even /20))
            for i in range(even, even + 20):
                x = data[i][0:2]
                label = labels[i]
                x = torch.Tensor([x]).reshape((2))
                label = torch.Tensor([label]).reshape((1, 1))
                output = bnn.forward(x)
                if j % 2 == 0:
                    opt.zero_grad()
                    error = loss(output, label)
                    error.backward()
                    opt.step()
                else:
                    # opt.zero_grad()
                    embedding_opt.zero_grad()
                    error = loss(output, label)
                    error.backward()
                    embedding_opt.step()
                    # opt.step()

            embedding_list = store_embedding_back_bnn(bnn, embedding_list, int(even / 20))

        for j in range(5):
            # chose an even schedule
            odd = int(np.random.choice(even_lambdas)) + 20
            load_in_embedding_bnn(bnn, embedding_list, int(odd / 20))
            for i in range(odd, odd+20):
                x = data[i][0:2]
                label = labels[i]
                x = torch.Tensor([x]).reshape((2))
                label = torch.Tensor([label]).reshape((1, 1))
                output = bnn.forward(x)
                if j % 2 == 0:
                    opt.zero_grad()
                    error = loss(output, label)
                    error.backward()
                    opt.step()
                else:
                    # opt.zero_grad()
                    embedding_opt.zero_grad()
                    error = loss(output, label)
                    error.backward()
                    embedding_opt.step()
                    # opt.step()
            embedding_list = store_embedding_back_bnn(bnn, embedding_list, int(odd / 20))


    test_data, test_labels = create_dual_neuron_dataset(20)
    print(bnn.state_dict())
    avg_loss = 0
    test_embedding_list = [torch.ones(1, 1) * 1/2 for _ in range(20)]
    embedding_opt = torch.optim.SGD(bnn.EmbeddingList[0].parameters(), lr=.1)
    counter = 0
    for i in range(20 * 20):
        load_in_embedding_bnn(bnn, test_embedding_list, int(i / 20))
        x = test_data[i][0:2]
        label = test_labels[i]
        x = torch.Tensor([x]).reshape((2))
        label = torch.Tensor([label]).reshape((1, 1))
        output = bnn.forward(x)

        error = loss(output, label)
        print('output is ', output)
        print('label is ', label)
        print('error is ', error.item())
        avg_loss += error.item()
        if error.item() < .05:
            counter +=1
        if error.item() > .05:
            flag = False
            tracker = 0
            while not flag:
                embedding_opt.zero_grad()
                error.backward()
                embedding_opt.step()
                output = bnn.forward(x)
                error = loss(output, label)
                tracker += 1
                if tracker > 100:
                    flag = True
                if error.item() < .1:
                    flag = True
        test_embedding_list = store_embedding_back_bnn(bnn, test_embedding_list, int(i / 20))


    print(test_embedding_list)
    print(embedding_list)
    avg_loss /= 400
    print('avg loss is', avg_loss)
    print ('accuracy', counter/400)
























# omegas, dataset = create_dataset()
# bnn = basic_bnn_net()
# params = list(bnn.parameters())
# del params[6]
# opt = torch.optim.Adam(params, lr=.0001)
# embedding_opt = torch.optim.Adam(bnn.EmbeddingList.parameters(), lr=.01)
# loss = nn.MSELoss()
# epochs = 50
# embedding_list = [torch.ones(1, 1) * 1/2 for _ in range(10)]
# for epoch in range(epochs):
#     for i in range(1000):
#         x = dataset[i]
#         label = x * omegas[i]
#         x = torch.Tensor([x]).reshape((1, 1))
#         label = torch.Tensor([label]).reshape((1, 1))
#         load_in_embedding_bnn(bnn, embedding_list, int(i / 100))
#         output = bnn.forward(x)
#
#         error = loss(output, label)
#         embedding_opt.zero_grad()
#         error.backward()
#         embedding_opt.step()
#         print('embedding: ', bnn.state_dict())
#
#         embedding_list = store_embedding_back_bnn(bnn, embedding_list, int(i /100))
#
#     for i in range(1000):
#         x = dataset[i]
#         label = x * omegas[i]
#         x = torch.Tensor([x]).reshape((1,1))
#         load_in_embedding_bnn(bnn, embedding_list, int(i / 100))
#         label = torch.Tensor([label]).reshape((1,1))
#         output = bnn.forward(x)
#         error = loss(output, label)
#         opt.zero_grad()
#         error.backward()
#         opt.step()
#         print('embedding: ', bnn.state_dict())
#         embedding_list =store_embedding_back_bnn(bnn, embedding_list, int(i /100))
#
#
#
# omegas_test, dataset_test = create_dataset()
# test_embedding_list = [torch.ones(1, 1) * 1/2 for _ in range(10)]
# embedding_opt = torch.optim.SGD(bnn.EmbeddingList.parameters(), lr=.8)
# for epoch in range(epochs):
#     for i in range(1000):
#         x = dataset_test[i]
#         label = x * omegas_test[i]
#         x = torch.Tensor([x]).reshape((1, 1))
#         label = torch.Tensor([label]).reshape((1, 1))
#         load_in_embedding_bnn(bnn, test_embedding_list, int(i / 100))
#         output = bnn.forward(x)
#
#         error = loss(output, label)
#         if error.item() > .05:
#             flag = False
#             tracker = 0
#             while not flag:
#                 output = bnn.forward(x)
#                 error = loss(output, label)
#                 embedding_opt.zero_grad()
#                 error.backward()
#                 embedding_opt.step()
#
#                 tracker += 1
#                 if tracker > 500:
#                     flag = True
#                 if error.item() < .05:
#                     flag = True
#         print('embedding: ', bnn.state_dict())
#         test_embedding_list = store_embedding_back_bnn(bnn, test_embedding_list, int(i / 100))
