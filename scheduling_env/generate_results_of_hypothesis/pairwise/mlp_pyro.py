import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.utils.data.dataloader as dataloader

device = torch.device("cpu")

log_softmax = nn.LogSoftmax(dim=1).to(device)
softplus = nn.Softplus().to(device)

batch_size, nx, nh, ny = 128, 28 * 28, 1024, 10

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
        self.l1 = torch.nn.Linear(nx, nh)
        self.l2 = torch.nn.Linear(nh, ny)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        h = self.relu(self.l1(x.view((-1, nx))))
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
        'l2.weight': normal(ny, nh), 'l2.bias': normal(ny)}
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
        'l2.weight': vnormal("W2", ny, nh), 'l2.bias':vnormal("b2", ny)}
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

train = MNIST("MNIST", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]), )
test = MNIST("MNIST", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]), )
dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=1, pin_memory=False)
train_loader = dataloader.DataLoader(train, **dataloader_args)
test_loader = dataloader.DataLoader(test, **dataloader_args)
num_epochs = 5
num_samples = 10


## Inference

# We can now launch the inference.

inference = SVI(model, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())

print('Inference')
for epoch in range(num_epochs):
    for j, (imgs, lbls) in enumerate(train_loader, 0):
        loss = inference.step(imgs.to(device), lbls.to(device))


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
correct = 0
total = 0
for j, data in enumerate(test_loader):
    images, labels = data
    predicted = predict(images)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print("accuracy: %d %%" % (100 * correct / total))

# That's it!

# You can now export the following environment
# variables with your Watson Machine Learning credentials:
# export ML_ENV=xxxxxxxxxxxxxxx
# export ML_INSTANCE=xxxxxxxxxxxxxxx
# export ML_USERNAME=xxxxxxxxxxxxxxx
# export ML_PASSWORD=xxxxxxxxxxxxxxx

# and run:
# zip code.zip mlp_pyro.py
# bx ml code.zip manifest.yml

# where code.zip is an archive containing all the python source files
# (e.g., edward.py, data-loaders, etc...). This command returns an id
# (e.g., training-xxxxxxxxx) that you can use to monitor the runnning
# job:
# bx ml monitor training-runs training-xxxxxxxxx
