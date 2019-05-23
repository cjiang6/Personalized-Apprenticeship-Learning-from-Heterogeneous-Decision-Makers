from base_testing_environment.using_bayes import create_simple_classification_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)
np.random.seed(50)  

num_schedules = 5
x_data, y = create_simple_classification_dataset(num_schedules, homog=True)

x = []
for each_ele in x_data:
    x.append(each_ele[2])

x = torch.Tensor(x).reshape(-1,1)
y = torch.Tensor(y).reshape((-1,1))
plt.scatter(x, y)
plt.title('Toy data datapoints')
plt.show()
print('Toy data datapoints')



class standard_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 16)
        self.l2 = nn.Linear(16, 16)
        self.l3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.l3(F.sigmoid(self.l2(F.sigmoid(self.l1(x)))))
        return x


# initialization of our standard neural network
net1 = standard_MLP()

# use of a Mean Square Error loss to evaluate the network because we are in a regression problem
criterion = nn.MSELoss()

# use of stochastic gradient descent as our optimizer
optimizer = optim.SGD(net1.parameters(), lr=0.01)

# number of times we are looping over our data to train the network
epochs = 40000

for epoch in range(epochs):
    optimizer.zero_grad()  # zero the gradient buffers
    output = net1(x)  # pass the data forward
    loss = criterion(output, y)  # evaluate our performance
    if epoch % 5000 == 0:
        print("epoch {} loss: {}".format(epoch, loss))
    loss.backward()  # calculates the gradients
    optimizer.step()  # updates weigths of the network

x_test = torch.linspace(-5, 5, 100).reshape(-1, 1)
predictions = net1(x_test)
plt.plot(x_test.numpy(), predictions.detach().numpy(), label='nn predictions')
plt.scatter(x, y, label='true values')
plt.title('Standard neural networks can overfit...')
plt.legend()
plt.show()


class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """

    def __init__(self, input_features, output_features, prior_var=1.):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        # initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        # initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0, prior_var)

    def forward(self, input):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape)
        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(input, self.w, self.b)


class MLP_BBB(nn.Module):
    def __init__(self, hidden_units, noise_tol=.1, prior_var=1.):
        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()
        self.hidden = Linear_BBB(1, hidden_units, prior_var=prior_var)
        self.out = Linear_BBB(hidden_units, 1, prior_var=prior_var)
        self.noise_tol = noise_tol  # we will use the noise tolerance to calculate our likelihood

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron
        x = torch.sigmoid(self.hidden(x))
        x = self.out(x)
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post

    def sample_elbo(self, input, target, samples):
        # we calculate the negative elbo, which will be our loss function
        # initialize tensors
        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input).reshape(-1)  # make predictions
            log_priors[i] = self.log_prior()  # get log prior
            log_posts[i] = self.log_post()  # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum()  # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss


net = MLP_BBB(32, prior_var=10)

optimizer = optim.Adam(net.parameters(), lr=.1)

epochs = 2000
for epoch in range(epochs):  # loop over the dataset multiple times
    optimizer.zero_grad()
    # forward + backward + optimize
    loss = net.sample_elbo(x, y, 1)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('epoch: {}/{}'.format(epoch + 1, epochs))
        print('Loss:', loss.item())
print('Finished Training')

# samples is the number of "predictions" we make for 1 x-value.
samples = 100
x_tmp = torch.linspace(-5, 5, 100).reshape(-1, 1)
y_samp = np.zeros((samples, 100))
for s in range(samples):
    y_tmp = net(x_tmp).detach().numpy()
    y_samp[s] = y_tmp.reshape(-1)
plt.plot(x_tmp.numpy(), np.mean(y_samp, axis=0), label='Mean Posterior Predictive')
plt.fill_between(x_tmp.numpy().reshape(-1), np.percentile(y_samp, 2.5, axis=0), np.percentile(y_samp, 97.5, axis=0), alpha=0.25, label='95% Confidence')
plt.legend()
plt.scatter(x, y)
plt.title('Posterior Predictive')
plt.show()


### Classification

batch_size = 1 # could also be 20
x_data_test, y_test = create_simple_classification_dataset(num_schedules, homog=True)

x_test = []
for each_ele in x_data_test:
    x_test.append(each_ele[2])

x_test = torch.Tensor(x_test).reshape(-1,1)
y_test = torch.Tensor(y_test).reshape((-1,1))

class Classifier_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.h1 = nn.Linear(in_dim, hidden_dim)
        self.h2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.log_softmax(self.out(x))
        return x


input_size = 1  # Just the x dimension
hidden_size = 10  # The number of nodes at the hidden layer
num_classes = 2  # The number of output classes. In this case, from 0 to 1
learning_rate = 1e-3  # The speed of convergence

MLP = Classifier_MLP(in_dim=input_size, hidden_dim=hidden_size, out_dim=num_classes)
optimizer = torch.optim.Adam(MLP.parameters(), lr=learning_rate)



epochs = 10
schedule_starts = np.linspace(0, 20 * (num_schedules-1), num=num_schedules)
for epoch in range(epochs):  # loop over the dataset multiple times
    # for batch, (x_train, y_train) in enumerate(train_loader):
    for i in range(num_schedules):
        chosen_schedule_start = int(np.random.choice(schedule_starts))
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            optimizer.zero_grad()
            pred = MLP(x[each_t])
            loss = F.cross_entropy(pred.reshape(1,2), y[each_t].long())
            loss.backward()
            optimizer.step()

    learning_rate /= 1.1
    test_losses, test_accs = [], []
    # for i, (x_test, y_test) in enumerate(test_loader):
    for i in range(num_schedules):
        chosen_schedule_start = int(schedule_starts[i])
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            optimizer.zero_grad()
            pred = MLP(x_test[each_t])
            loss = F.cross_entropy(pred.reshape(1,2), y_test[each_t].long())
            acc = (pred.argmax(dim=-1) == y_test[each_t].item()).to(torch.float32).mean()
            test_losses.append(loss.item())
            test_accs.append(acc.mean().item())
    print('Loss: {}, Accuracy: {}'.format(np.mean(test_losses), np.mean(test_accs)))
print('Finished Training')


class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """

    def __init__(self, input_features, output_features, prior_var=6):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        super().__init__()
        # set dim
        self.input_features = input_features
        self.output_features = output_features

        # initialize weight params
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-0.6, 0.6))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-6, -6))

        # initialize bias params
        self.b_mu = nn.Parameter(torch.zeros(output_features).uniform_(-0.6, 0.6))
        self.b_rho = nn.Parameter(torch.zeros(output_features).uniform_(-6, -6))

        # initialize weight samples
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape)
        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        # initialize prior distribution
        self.prior = torch.distributions.Normal(0, prior_var)

    def forward(self, input):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape)
        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        # record prior
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record variational_posterior
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(input, self.w, self.b)


class Classifier_BBB(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.h1 = Linear_BBB(in_dim, hidden_dim)
        self.h2 = Linear_BBB(hidden_dim, hidden_dim)
        self.out = Linear_BBB(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        # x = x.view(-1, 28*28)
        x = torch.sigmoid(self.h1(x))
        x = torch.sigmoid(self.h2(x))
        x = F.log_softmax(self.out(x))
        return x

    def log_prior(self):
        return self.h1.log_prior + self.h2.log_prior + self.out.log_prior

    def log_post(self):
        return self.h1.log_post + self.h2.log_post + self.out.log_post

    def sample_elbo(self, input, target, samples):
        outputs = torch.zeros(samples, target.shape[0], self.out_dim)
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        # log_likes = torch.zeros(samples)
        for i in range(samples):
            outputs[i] = self(input)
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()
            # log_likes[i] = torch.log(outputs[i, torch.arange(outputs.shape[1]), target]).sum(dim=-1)
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        # log_likes = F.nll_loss(outputs.mean(0), target, size_average=False)
        log_likes = F.nll_loss(outputs.mean(0), target, reduction='sum')
        loss = (log_post - log_prior) / num_batches + log_likes
        return loss, outputs





input_size = 1  # Just the x dimension
hidden_size = 10  # The number of nodes at the hidden layer
num_classes = 2  # The number of output classes. In this case, from 0 to 1
learning_rate = 1e-3  # The speed of convergence
num_batches = num_schedules * 20

classifier = Classifier_BBB(in_dim=input_size, hidden_dim=hidden_size, out_dim=num_classes)
optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)


epochs = 5000
schedule_starts = np.linspace(0, 20 * (num_schedules-1), num=num_schedules)
for epoch in range(epochs):  # loop over the dataset multiple times
    # for batch, (x_train, y_train) in enumerate(train_loader):
    for i in range(num_schedules):
        chosen_schedule_start = int(np.random.choice(schedule_starts))
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            optimizer.zero_grad()
            loss, _ = classifier.sample_elbo(x[each_t], y[each_t].long(), 1)
            loss.backward()
            optimizer.step()

    learning_rate /= 1.1
    test_losses, test_accs = [], []
    # for i, (x_test, y_test) in enumerate(test_loader):
    for i in range(num_schedules):
        chosen_schedule_start = int(schedule_starts[i])
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            optimizer.zero_grad()
            test_loss, test_pred = classifier.sample_elbo(x_test[each_t], y_test[each_t].long(), 5)
            acc = (test_pred.argmax(dim=-1) == y_test[each_t].item()).to(torch.float32).mean() # maybe this is where err comes from
            test_losses.append(test_loss.item())
            test_accs.append(acc.mean().item())
    print('Loss: {}, Accuracy: {}'.format(np.mean(test_losses), np.mean(test_accs)))
print('Finished Training')


















