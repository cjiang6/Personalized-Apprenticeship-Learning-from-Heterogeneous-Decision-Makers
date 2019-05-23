import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pyro
from pyro.distributions import Normal, Categorical, Bernoulli
from pyro.infer import SVI, Trace_ELBO, RenyiELBO, JitTrace_ELBO
from pyro.optim import Adam
from base_testing_environment.using_bayes import create_simple_classification_dataset
device = torch.device("cpu")

log_softmax = nn.LogSoftmax(dim=1).to(device)
softplus = nn.Softplus().to(device)

batch_size, nx, nh, ny = 20, 3 * 1,3, 3

## Bayesian MLP in Pyro

# The main idea is to treat all the weights and biases of the
# network as random variables. We thus learn a complete distribution for
# each parameter of the network. These distributions can be used to
# measure the uncertainty associated to the ouptut of the network, which
# can be critical for decision-making systems. As in Edward, we will use
# *variational inference* to learn the distributions in Pyro.

# In short we need to define two main components:

# 1. The probabilistic model: a MLP where all weights and biases
# are treated as random variable
# 2. A family of guide distributions for the Variational Inference

# The corresponding Pyro code is the following:


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(3, 3)
        self.l1a = torch.nn.Linear(3, 3)
        self.l2 = torch.nn.Linear(3, 2)
        self.relu = torch.nn.ReLU()
        bayes_embed = torch.Tensor([1]).reshape(1)
        bayes_embed.requires_grad = False
        self.bayesian_embedding = nn.Parameter(bayes_embed)

    def forward(self, x,n=20):
        h = self.relu(self.l1(torch.cat([torch.Tensor(x),torch.Tensor(self.bayesian_embedding.expand(n).reshape(n,1))],dim=1)))
        h = self.relu(self.l1a(h.view((-1, nh))))
        yhat = self.l2(h)
        return yhat

mlp = MLP().to(device)


# Model

def normal(*shape):
    loc = torch.zeros(*shape).to(device)
    scale = torch.ones(*shape).to(device)
    return Normal(loc, scale)

def model(imgs, lbls):
    priors = {
        'l1.weight': normal(nh, nx), 'l1.bias': normal(nh),
        'l1a.weight': normal(nh, nh), 'l1a.bias': normal(nh),
        'l2.weight': normal(ny, nh), 'l2.bias': normal(ny),
        'bayesian_embedding': Bernoulli(probs=.5)}

    lifted_module = pyro.random_module("mlp", mlp, priors)
    lifted_reg_model = lifted_module()
    lhat = log_softmax(lifted_reg_model(imgs))
    pyro.sample("obs", Categorical(logits=lhat), obs=lbls)


# Inference Guide

def vnormal(name, *shape):
    loc = pyro.param(name+"m", torch.randn(*shape, requires_grad=True, device=device))
    scale = pyro.param(name+"s", torch.randn(*shape, requires_grad=True, device=device))
    return Normal(loc, softplus(scale))

def guide(imgs, lbls):
    dists = {
        'l1.weight': vnormal("W1", nh, nx), 'l1.bias': vnormal("b1", nh),
        'l1a.weight': vnormal("W3", nh, nh), 'l1a.bias': vnormal("b3", nh),
        'l2.weight': vnormal("W2", ny, nh), 'l2.bias':vnormal("b2", ny),
        'bayesian_embedding': vnormal('em', 1)}
    lifted_module = pyro.random_module("mlp", mlp, dists)
    return lifted_module()

# The MLP network is defined in PyTorch. In the model, we first define
# the prior distributions for all the weights and biases and then lift
# the MLP definition from concrete to probabilistic using the
# `pyro.random_module` function.  The result `yhat` parameterizes a
# categorical distribution over the possible labels for an image
# `x`. Note the `pyro.observe` statement that will match the prediction
# of the network `yhat` with the known label `y` during the inference.

# The guide defines the family of distributions used for Variational
# Inference. In our case all the parameters follow a normal
# distribution. Note the use of `Variable` to define the parameters of
# the family (here the means and scale of the normal
# distribution). After the training these variables contain the
# parameters of the distribution that is the closest to the true
# posterior.

## Data

# Before starting the training, let us import the MNIST dataset.

# train = MNIST("MNIST", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]), )
# test = MNIST("MNIST", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]), )
# dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=1, pin_memory=False)
# train_loader = dataloader.DataLoader(train, **dataloader_args)
# test_loader = dataloader.DataLoader(test, **dataloader_args)
num_epochs = 500
num_samples = 100


## Inference

# We can now launch the inference.

inference = SVI(model, guide, Adam({"lr": 0.001}), loss=RenyiELBO(alpha=.5))
num_schedules = 50
data, labels = create_simple_classification_dataset(num_schedules)
schedule_starts = np.linspace(0, 20 * (num_schedules-1), num=num_schedules)
not_first_time = False
distributions = [np.array([.5, .1], dtype=float) for _ in range(num_schedules)]  # each one is mean, sigma

print('Inference')
for epoch in range(num_epochs):
    # for j, (imgs, lbls) in enumerate(train_loader, 0):
    #     loss = inference.step(imgs.to(device), lbls.to(device))
    for _ in range(num_schedules):
        x_data = []
        y_data = []
        chosen_schedule_start = int(np.random.choice(schedule_starts))
        schedule_num = int(chosen_schedule_start / 20)
        if not_first_time:
            pyro.get_param_store().get_state()['params']['emm']=Variable(torch.Tensor([distributions[schedule_num][0]]),requires_grad=True)
            pyro.get_param_store().get_state()['params']['ems']=Variable(torch.Tensor([distributions[schedule_num][1]]),requires_grad=True)
            # print(pyro.get_param_store().get_state()['params']['emm'])
        else:
            not_first_time = True
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            x = data[each_t][2:]
            x_data.append(x)
            # noinspection PyArgumentList
            x = torch.Tensor([x]).reshape((2))

            label = labels[each_t]
            y_data.append(label)
            # noinspection PyArgumentList
            label = torch.Tensor([label]).reshape(1)
            label = Variable(label).long()

        loss = inference.step(x_data, torch.Tensor(y_data).long())
        distributions[schedule_num][0] = pyro.get_param_store()['emm'].item()
        distributions[schedule_num][1] = pyro.get_param_store()['ems'].item()
print(distributions)
## Prediction

# When the training is complete, we can sample the guide containing the
# posterior distribution multiple times to obtain a set of MLPs. We can
# then combine the predictions of all the MLPs to compute a prediction.

def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return np.argmax(mean, axis=1)







print('Prediction')



acc = 0
for i in range(num_schedules):
    chosen_schedule_start = int(schedule_starts[i])
    schedule_num = int(chosen_schedule_start / 20)
    pyro.get_param_store().get_state()['params']['emm'] = Variable(torch.Tensor([distributions[schedule_num][0]]), requires_grad=True)
    pyro.get_param_store().get_state()['params']['ems'] = Variable(torch.Tensor([distributions[schedule_num][1]]), requires_grad=True)
    x_data = []
    y_data = []
    for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
        # at each timestep you what to resample the embedding

        x = data[each_t][2:]
        x_data.append(x)
        x = torch.Tensor(x).reshape((2))
        # output = predict(x)

        label = labels[each_t]
        y_data.append(label)
        label = torch.Tensor([label]).reshape(1)
        label = Variable(label).long()

    output = predict(x_data)
    for i,each_output in enumerate(output):

        if output[i].item() == labels[i][0]:
            acc += 1


print("accuracy: %d %%" % (100 * acc / (20*num_schedules)))
correct = 0
total = 0
# for j, data in enumerate(test_loader):
#     images, labels = data
#     predicted = predict(images)
#     total += labels.size(0)
#     correct += (predicted == labels).sum().item()
# print("accuracy: %d %%" % (100 * correct / total))

# That's it!


