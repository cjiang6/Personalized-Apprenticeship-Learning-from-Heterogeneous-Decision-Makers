"""
experimenting with new BNN
"""
import torch
import torch.nn as nn
import pyro
import numpy as np
import sys
sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')
from pyro.optim import SGD, Adam
from pyro.distributions import Normal, Categorical, Bernoulli, BetaBinomial, Binomial
from pyro.infer import SVI, Trace_ELBO, RenyiELBO, JitTrace_ELBO
from pyro.contrib.autoguide import AutoDiagonalNormal
from base_testing_environment.using_bayes import create_simple_classification_dataset
from torch.autograd import Variable
import pdb
pyro.set_rng_seed(101)


class BNN(nn.Module):
    def __init__(self):
        super(BNN, self).__init__()
        self.fc1 = nn.Linear(4, 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4, 1)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(2, 2)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(2, 2)
        bayes_embed = torch.Tensor([1]).reshape(1)
        bayes_embed.requires_grad = False
        self.bayesian_embedding = nn.Parameter(bayes_embed)
        # self.factor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        """

        :param x: lets specify x is made of lambda, z, x
        :return:
        """
        x = torch.cat([x, self.bayesian_embedding.reshape(1), 1-self.bayesian_embedding.reshape(1)], dim=0)
        x1 = self.fc1(x.reshape((4)))
        x1 = self.relu1(x1)
        x2 = self.fc2(x.reshape((4)))
        x2 = self.relu2(x2)
        x3 = self.fc3(x.reshape((4)))
        x3 = self.relu3(x3)
        x = self.fc4(torch.cat([x1]))
        x = self.relu(x)
        x = self.fc5(x)
        return x


net = BNN()
softmax = nn.LogSoftmax(dim=1)


def model(x_data, y_data):
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))
    # fc2w_prior = Normal(loc=torch.zeros_like(net.fc2.weight), scale=torch.ones_like(net.fc2.weight))
    # fc2b_prior = Normal(loc=torch.zeros_like(net.fc2.bias), scale=torch.ones_like(net.fc2.bias))
    # fc3w_prior = Normal(loc=torch.zeros_like(net.fc3.weight), scale=torch.ones_like(net.fc3.weight))
    # fc3b_prior = Normal(loc=torch.zeros_like(net.fc3.bias), scale=torch.ones_like(net.fc3.bias))
    fc4w_prior = Normal(loc=torch.zeros_like(net.fc4.weight), scale=torch.ones_like(net.fc4.weight))
    fc4b_prior = Normal(loc=torch.zeros_like(net.fc4.bias), scale=torch.ones_like(net.fc4.bias))
    fc5w_prior = Normal(loc=torch.zeros_like(net.fc5.weight), scale=torch.ones_like(net.fc5.weight))
    fc5b_prior = Normal(loc=torch.zeros_like(net.fc5.bias), scale=torch.ones_like(net.fc5.bias))
    # embedding_prior =Bernoulli(probs=.5) # Normal(loc=torch.zeros_like(net.bayesian_embedding),scale=torch.ones_like(net.bayesian_embedding)) # Normal(loc=torch.zeros_like(net.bayesian_embedding),scale=torch.ones_like(net.bayesian_embedding)) #Binomial(probs=.5)

    priors = {'fc1.weight': fc1w_prior,
              'fc1.bias': fc1b_prior,
              'fc4.weight': fc4w_prior,
              'fc4.bias': fc4b_prior,
              'fc5.weight': fc5w_prior,
              'fc5.bias': fc5b_prior}
              # 'bayesian_embedding': embedding_prior}

    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    lhat = softmax(lifted_reg_model(x_data).reshape(1, 2))

    # we tell pyro that output is categorical in nature
    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)


softplus = torch.nn.Softplus()  # why do we need a soft relu
sig = torch.nn.Sigmoid()

def guide(x_data, y_data):
    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    # First layer bias distribution priors
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

    # # Second layer weight distribution priors
    # fc2w_mu = torch.randn_like(net.fc2.weight)
    # fc2w_sigma = torch.randn_like(net.fc2.weight)
    # fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
    # fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
    # fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)
    # # Second layer bias distribution priors
    # fc2b_mu = torch.randn_like(net.fc2.bias)
    # fc2b_sigma = torch.randn_like(net.fc2.bias)
    # fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
    # fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
    # fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)
    #
    # # Third layer weight distribution priors
    # fc3w_mu = torch.randn_like(net.fc3.weight)
    # fc3w_sigma = torch.randn_like(net.fc3.weight)
    # fc3w_mu_param = pyro.param("fc3w_mu", fc3w_mu)
    # fc3w_sigma_param = softplus(pyro.param("fc3w_sigma", fc3w_sigma))
    # fc3w_prior = Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param)
    # # Third layer bias distribution priors
    # fc3b_mu = torch.randn_like(net.fc3.bias)
    # fc3b_sigma = torch.randn_like(net.fc3.bias)
    # fc3b_mu_param = pyro.param("fc3b_mu", fc3b_mu)
    # fc3b_sigma_param = softplus(pyro.param("fc3b_sigma", fc3b_sigma))
    # fc3b_prior = Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param)

    # Forth layer weight distribution priors
    fc4w_mu = torch.randn_like(net.fc4.weight)
    fc4w_sigma = torch.randn_like(net.fc4.weight)
    fc4w_mu_param = pyro.param("fc4w_mu", fc4w_mu)
    fc4w_sigma_param = softplus(pyro.param("fc4w_sigma", fc4w_sigma))
    fc4w_prior = Normal(loc=fc4w_mu_param, scale=fc4w_sigma_param)
    # Fourth layer bias distribution priors
    fc4b_mu = torch.randn_like(net.fc4.bias)
    fc4b_sigma = torch.randn_like(net.fc4.bias)
    fc4b_mu_param = pyro.param("fc4b_mu", fc4b_mu)
    fc4b_sigma_param = softplus(pyro.param("fc4b_sigma", fc4b_sigma))
    fc4b_prior = Normal(loc=fc4b_mu_param, scale=fc4b_sigma_param)

    # Fifth layer weight distribution priors
    fc5w_mu = torch.randn_like(net.fc5.weight)
    fc5w_sigma = torch.randn_like(net.fc5.weight)
    fc5w_mu_param = pyro.param("fc5w_mu", fc5w_mu)
    fc5w_sigma_param = softplus(pyro.param("fc5w_sigma", fc5w_sigma))
    fc5w_prior = Normal(loc=fc5w_mu_param, scale=fc5w_sigma_param)
    # Fifth layer bias distribution priors
    fc5b_mu = torch.randn_like(net.fc5.bias)
    fc5b_sigma = torch.randn_like(net.fc5.bias)
    fc5b_mu_param = pyro.param("fc5b_mu", fc5b_mu)
    fc5b_sigma_param = softplus(pyro.param("fc5b_sigma", fc5b_sigma))
    fc5b_prior = Normal(loc=fc5b_mu_param, scale=fc5b_sigma_param)


    # embedding_mu = torch.ones_like(net.bayesian_embedding)*.5
    # embedding_sigma = torch.randn_like(net.bayesian_embedding)
    # embedding_mu_param = pyro.param("embedding_mu", embedding_mu)
    # embedding_sigma_param = softplus(pyro.param("embedding_sigma", embedding_sigma))
    # embedding_prior =  Normal(loc=embedding_mu_param, scale=embedding_sigma_param)# Bernoulli(probs=embedding_mu_param) #)   # Bernoulli(probs=embedding_mu_param) # Normal(loc=embedding_mu_param, scale=embedding_sigma_param)        # Binomial(probs=embedding_mu_param)

    priors = {'fc1.weight': fc1w_prior,
              'fc1.bias': fc1b_prior,
              # 'fc2.weight': fc2w_prior,
              # 'fc2.bias': fc2b_prior,
              # 'fc3.weight': fc3w_prior,
              # 'fc3.bias': fc3b_prior,
              'fc4.weight': fc4w_prior,
              'fc4.bias': fc4b_prior,
              'fc5.weight': fc5w_prior,
              'fc5.bias': fc5b_prior}
              # 'bayesian_embedding': embedding_prior}
    # print(embedding_prior)
    lifted_module = pyro.random_module("module", net, priors)

    return lifted_module()


print(net.state_dict())
# guide = AutoDiagonalNormal(model)
optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO()) #Trace_ELBO doesn't like binomial

# optimization loop
num_schedules = 100
data, labels = create_simple_classification_dataset(num_schedules)
# bnn.set_weights_and_bias()

distributions = [np.array([.5, 1], dtype=float) for _ in range(num_schedules)]  # each one is mean, sigma
epochs = 2
schedule_starts = np.linspace(0, 1980, num=num_schedules)
pyro.clear_param_store()
not_first_time = False
for i in range(4):
    for epoch in range(epochs):
        for _ in range(num_schedules):
            # choose a schedule
            chosen_schedule_start = int(np.random.choice(schedule_starts))
            # chosen_schedule_start = 20 # uncomment if you want to test on schedule
            schedule_num = int(chosen_schedule_start / 20)
            # embedding_given_dis = get_embedding_given_dist(distributions[schedule_num])
            if not_first_time:
                pass
                # pyro.get_param_store().get_state()['params']['embedding_mu']=Variable(torch.Tensor([distributions[schedule_num][0]]),requires_grad=True)
                # pyro.get_param_store().get_state()['params']['embedding_sigma']=Variable(torch.Tensor([distributions[schedule_num][1]]),requires_grad=True)
            else:
                not_first_time = True
            for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
                x = data[each_t][2:]
                # noinspection PyArgumentList
                x = torch.Tensor([x]).reshape((2))
                label = labels[each_t]
                # noinspection PyArgumentList
                label = torch.Tensor([label]).reshape(1)
                label = Variable(label).long()

                loss = svi.step(x, label)
                # print(net.state_dict())
                # print(loss)
            # print(net.state_dict())
            # distributions[schedule_num][0] = pyro.get_param_store()['embedding_mu'].item()
            # distributions[schedule_num][1] = pyro.get_param_store()['embedding_sigma'].item()
    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))
    num_samples = 5
    print(distributions)

    def predict(x):
        sampled_models = [guide(None, None) for _ in range(num_samples)]
        yhats = [z(x).data for z in sampled_models]
        mean = torch.mean(torch.stack(yhats), 0)
        return torch.argmax(mean).item()


    num_schedules = 100
    # lmao = predict(torch.Tensor(test_data[2][2:]).reshape((2)))
    schedule_starts = np.linspace(0, 1980, num=100)
    total_acc = []


    def load_in_embedding(NeuralNet, embedding):
        curr_embedding = embedding
        curr_dict = NeuralNet.state_dict()
        curr_dict['bayesian_embedding'] = curr_embedding
        NeuralNet.load_state_dict(curr_dict)


    for i in range(num_schedules):
        # choose a schedule
        chosen_schedule_start = int(schedule_starts[i])
        # chosen_schedule_start = 20
        schedule_num = int(chosen_schedule_start / 20)
        # pyro.get_param_store().get_state()['params']['embedding_mu'] = Variable(torch.Tensor([distributions[schedule_num][0]]), requires_grad=True)
        # pyro.get_param_store().get_state()['params']['embedding_sigma'] = Variable(torch.Tensor([distributions[schedule_num][1]]), requires_grad=True)

        # svi = SVI(model, guide, optim, loss=RenyiELBO())
        acc = 0
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            # at each timestep you what to resample the embedding

            x = data[each_t][2:]

            x = torch.Tensor(x).reshape((2))
            output = predict(x)

            label = labels[each_t]
            label = torch.Tensor([label]).reshape(1)
            label = Variable(label).long()
            # print('output is ', output, ' label is ', label.item())
            if output == label.item():
                acc += 1

        total_acc.append(acc / 20)
        print('schedule accuracy: ', acc/20)
    print('mean is ', np.mean(total_acc))
    print('finite')



# from pyro.optim import Adam
#
# def per_param_callable(module_name, param_name):
#     if param_name == 'bayesian_embedding':
#         return {"lr": 0.010}
#     else:
#         return {"lr": 0.001}
#
# optimizer = Adam(per_param_callable)
