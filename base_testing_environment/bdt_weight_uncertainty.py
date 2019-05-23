from base_testing_environment.using_bayes import create_simple_classification_dataset
from base_testing_environment.bdt import Gaussian_ProLoNet
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

x = torch.Tensor(x).reshape(-1, 1)
y = torch.Tensor(y).reshape((-1,1))
# plt.scatter(x, y)
# plt.title('Toy data datapoints')
# plt.show()

net = Gaussian_ProLoNet(input_dim=1,
                        weights=None,
                        comparators=None,
                        leaves=4,
                        selectors=None,
                        output_dim=1,
                        is_value=True,
                        prior_var=10)
optimizer = optim.Adam(net.parameters(), lr=.1)

# epochs = 2000
# for epoch in range(epochs):  # loop over the dataset multiple times
#     optimizer.zero_grad()
#     # forward + backward + optimize
#     loss = net.sample_elbo(x, y, 1)
#     loss.backward()
#     optimizer.step()
#     if epoch % 10 == 0:
#         print('epoch: {}/{}'.format(epoch + 1, epochs))
#         print('Loss:', loss.item())
# print('Finished Training')

# # samples is the number of "predictions" we make for 1 x-value.
# samples = 100
# x_tmp = torch.linspace(-5, 5, 100).reshape(-1, 1)
# y_samp = np.zeros((samples, 100))
# for s in range(samples):
#     y_tmp = net(x_tmp).detach().numpy()
#     y_samp[s] = y_tmp.reshape(-1)
# plt.plot(x_tmp.numpy(), np.mean(y_samp, axis=0), label='Mean Posterior Predictive')
# plt.fill_between(x_tmp.numpy().reshape(-1), np.percentile(y_samp, 2.5, axis=0), np.percentile(y_samp, 97.5, axis=0), alpha=0.25, label='95% Confidence')
# plt.legend()
# plt.scatter(x, y)
# plt.title('Posterior Predictive')
# plt.show()


### Classification

batch_size = 1 # could also be 20
x_data_test, y_test = create_simple_classification_dataset(num_schedules, homog=True)

x_test = []
for each_ele in x_data_test:
    x_test.append(each_ele[2])

x_test = torch.Tensor(x_test).reshape(-1,1)
y_test = torch.Tensor(y_test).reshape((-1,1))

input_size = 1  # Just the x dimension
hidden_size = 10  # The number of nodes at the hidden layer
num_classes = 2  # The number of output classes. In this case, from 0 to 1
learning_rate = 1e-3  # The speed of convergence
num_batches = num_schedules * 20

classifier = Gaussian_ProLoNet(input_dim=input_size,
                               weights=None,
                               comparators=None,
                               leaves=4,
                               selectors=None,
                               output_dim=num_classes,
                               is_value=False,
                               prior_var=10,
                               num_batches=num_batches)
# classifier = Classifier_BBB(in_dim=input_size, hidden_dim=hidden_size, out_dim=num_classes)
optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)


epochs = 5000
schedule_starts = np.linspace(0, 20 * (num_schedules-1), num=num_schedules)
for epoch in range(epochs):  # loop over the dataset multiple times
    # for batch, (x_train, y_train) in enumerate(train_loader):
    for i in range(num_schedules):
        chosen_schedule_start = int(np.random.choice(schedule_starts))
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            optimizer.zero_grad()
            loss, _ = classifier.sample_elbo(x[each_t].unsqueeze(0), y[each_t].unsqueeze(0).long(), 1)
            loss.backward()
            optimizer.step()

    learning_rate /= 1.1
    test_losses, test_accs = [], []
    # for i, (x_test, y_test) in enumerate(test_loader):
    for i in range(num_schedules):
        chosen_schedule_start = int(schedule_starts[i])
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            optimizer.zero_grad()
            test_loss, test_pred = classifier.sample_elbo(x_test[each_t].unsqueeze(0), y_test[each_t].unsqueeze(0).long(), 5)
            acc = (test_pred.argmax(dim=-1) == y_test[each_t].item()).to(torch.float32).mean()
            test_losses.append(test_loss.item())
            test_accs.append(acc.mean().item())
    print('Loss: {}, Accuracy: {}'.format(np.mean(test_losses), np.mean(test_accs)))
print('Finished Training')


















