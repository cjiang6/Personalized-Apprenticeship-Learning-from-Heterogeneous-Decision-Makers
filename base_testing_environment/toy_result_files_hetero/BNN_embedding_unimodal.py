"""
BNN w/ unimodal embedding
This means all parameters have a gaussian upon them
"""

from base_testing_environment.toy_result_files_hetero.generate_environment import create_simple_classification_dataset
from base_testing_environment.utils.accuracy_measures import compute_specificity, compute_sensitivity

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)
np.random.seed(50)


# embedding module
class EmbeddingModule(nn.Module):
    """
    embedding class (allows us to access parameters directly)
    """

    def __init__(self, input_features, output_features, prior_var=6):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        super(EmbeddingModule, self).__init__()
        # set dim
        self.input_features = input_features
        self.output_features = output_features

        # initialize weight params
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-0.6, 0.6))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-6, -6))


        # initialize weight samples
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # initialize prior distribution
        self.prior = torch.distributions.Normal(0, prior_var)

    def forward(self):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon


        # record prior
        w_log_prior = self.prior.log_prob(self.w)
        self.log_prior = torch.sum(w_log_prior)

        # record variational_posterior
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum()

        return self.w


class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """

    def __init__(self, input_features, output_features, prior_var=6):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        super(Linear_BBB, self).__init__()
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
        super(Classifier_BBB, self).__init__()
        self.h1 = Linear_BBB(in_dim+2, hidden_dim)
        self.h2 = Linear_BBB(hidden_dim, hidden_dim)
        self.out = Linear_BBB(hidden_dim, out_dim)
        self.out_dim = out_dim
        self.embedding = EmbeddingModule(2,1)

    def forward(self, x):
        # x = x.view(-1, 28*28)
        w = self.embedding()
        x = torch.cat([x, w.reshape(2)], dim=0)
        x = torch.sigmoid(self.h1(x))
        x = torch.sigmoid(self.h2(x))
        x = F.log_softmax(self.out(x))
        return x

    def log_prior(self):
        return self.h1.log_prior + self.h2.log_prior + self.embedding.log_prior + self.out.log_prior

    def log_post(self):
        return self.h1.log_post + self.h2.log_post + self.embedding.log_prior + self.out.log_post

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

    def set_bayesian_embedding(self, embedding_means, embedding_stds):
        """
        sets embedding into BNN
        :param embedding:
        :return:
        """
        self.embedding.w_mu.requires_grad = False
        for n, i in enumerate(embedding_means):
            self.embedding.w_mu[0][n].fill_(i)
        self.embedding.w_mu.requires_grad = True
        self.embedding.w_rho.requires_grad = False
        for n, i in enumerate(embedding_stds):
            self.embedding.w_rho[0][n].fill_(i)

        self.embedding.w_rho.requires_grad = True


    def get_bayesian_embedding(self):
        """
        gets embedding inside BNN
        :return:
        """
        return self.embedding.w_mu.data, self.embedding.w_rho.data


# generate training data
num_schedules = 50
x_data, y = create_simple_classification_dataset(num_schedules, train=True)

x = []
for each_ele in x_data:
    x.append(each_ele[2:])

x = torch.Tensor(x).reshape(-1, 2)
y = torch.Tensor(y).reshape((-1, 1))
print('Toy problem generated, and data cleaned')

x_data_cv, y_cv = create_simple_classification_dataset(10, cv=True)

x_cv = []
for each_ele in x_data_cv:
    x_cv.append(each_ele[2:])

x_cv = torch.Tensor(x_cv).reshape(-1, 2)
y_cv = torch.Tensor(y_cv).reshape((-1, 1))

mean_distributions = [np.zeros(2) for _ in range(num_schedules)]
cv_mean_distributions = [np.zeros(2)  for _ in range(10)]

std_distributions = [np.ones(2) * -2 for _ in range(num_schedules)]
cv_std_distributions = [np.ones(2) * -2 for _ in range(10)]

input_size = 2  # Just the x dimension
hidden_size = 10  # The number of nodes at the hidden layer
num_classes = 2  # The number of output classes. In this case, from 0 to 1
learning_rate = 1e-3  # The speed of convergence
num_batches = num_schedules * 20

classifier = Classifier_BBB(in_dim=input_size, hidden_dim=hidden_size, out_dim=num_classes)
# optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD([{'params': list(classifier.parameters())[:-2]}, {'params': classifier.embedding.parameters(), 'lr': .1}], lr=learning_rate)
solo_embedding_optimizer = torch.optim.SGD([{'params': classifier.embedding.parameters()}], lr=.3)


epochs = 50
schedule_starts = np.linspace(0, 20 * (num_schedules-1), num=num_schedules)
for epoch in range(epochs):  # loop over the dataset multiple times
    # for batch, (x_train, y_train) in enumerate(train_loader):
    for i in range(num_schedules):
        chosen_schedule_start = int(np.random.choice(schedule_starts))
        schedule_num = int(chosen_schedule_start / 20)

        classifier.set_bayesian_embedding(list(mean_distributions[schedule_num]), list(std_distributions[schedule_num]))

        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            print(classifier.embedding.state_dict())
            optimizer.zero_grad()
            loss, _ = classifier.sample_elbo(x[each_t], y[each_t].long(), 1)
            loss.backward()
            optimizer.step()

        a, b = classifier.get_bayesian_embedding()

        mean_distributions[schedule_num], std_distributions[schedule_num] = list(a.data.detach().numpy()[0]), list(b.data.detach().numpy()[0])  # very ugly



    learning_rate /= 1.1
    cv_losses, cv_accs = [], []
    # for i, (x_cv, y_cv) in enumerate(cv_loader):
    for i in range(10):
        chosen_schedule_start = int(schedule_starts[i])
        schedule_num = int(chosen_schedule_start / 20)
        classifier.set_bayesian_embedding(list(cv_mean_distributions[schedule_num]), list(cv_std_distributions[schedule_num]))

        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            print('cv: ', classifier.embedding.state_dict())
            solo_embedding_optimizer.zero_grad()
            cv_loss, cv_pred = classifier.sample_elbo(x_cv[each_t], y_cv[each_t].long(), 5)
            cv_loss.backward()
            solo_embedding_optimizer.step()
            acc = (cv_pred.argmax(dim=-1) == y_cv[each_t].item()).to(torch.float32).mean()
            cv_losses.append(cv_loss.item())
            cv_accs.append(acc.mean().item())
    print('Loss: {}, Accuracy: {}'.format(np.mean(cv_losses), np.mean(cv_accs)))
print('Finished Training')

## Real Test

x_data_test, y_test, percent_of_zeros = create_simple_classification_dataset(50, True)

x_test = []
for each_ele in x_data_test:
    x_test.append(each_ele[2:])


x_test = torch.Tensor(x_test).reshape(-1, 2)
y_test = torch.Tensor(y_test).reshape((-1, 1))
test_losses, test_accs = [], []
per_schedule_test_losses, per_schedule_test_accs = [], []
preds, actual = [[] for _ in range(50)], [[] for _ in range(50)]
for i in range(50):
    chosen_schedule_start = int(schedule_starts[i])
    for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
        optimizer.zero_grad()
        test_loss, test_pred = classifier.sample_elbo(x_test[each_t], y_test[each_t].long(), 5)
        mean_test_pred = test_pred.mean(dim=0)
        preds[i].append(mean_test_pred.argmax(dim=-1).item())
        actual[i].append(y_test[each_t].item())
        acc = (mean_test_pred.argmax(dim=-1) == y_test[each_t].item()).to(torch.float32).mean()
        test_losses.append(test_loss.item())
        test_accs.append(acc.mean().item())
    per_schedule_test_accs.append(np.mean(test_accs))

sensitivity, specificity = compute_sensitivity(preds, actual), compute_specificity(preds, actual)

print('Loss: {}, Accuracy: {}'.format(np.mean(test_losses), np.mean(test_accs)))
print('per sched accuracy: ', np.mean(per_schedule_test_accs))
print('mean sensitivity: ', sensitivity, ', mean specificity: ', specificity)
# Compute sensitivity and specificity (ideally these should be very high)
# file = open('heterogeneous_toy_env_results.txt', 'a')
# file.write('BNN w/ unimodal embedding: mean: ' +
#            str(np.mean(per_schedule_test_accs)) +
#            ', std: ' + str(np.std(per_schedule_test_accs)) +
#             ', sensitivity: ' + str(sensitivity) + ', specificity: '+ str(specificity) +
#            ', Distribution of Class: 0: ' + str(percent_of_zeros) + ', 1: ' + str(1 - percent_of_zeros) +
#            '\n')
# file.close()
