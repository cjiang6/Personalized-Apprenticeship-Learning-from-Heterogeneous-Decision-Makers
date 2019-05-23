"""
Base NN implementation evaluating train and test performance on a homogeneous dataset
created on May 17, 2019 by Ghost
"""
from base_testing_environment.toy_result_files_homo.generate_environment import create_simple_classification_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)  # ensures repeatability
np.random.seed(50)



num_schedules = 50
x_data, y = create_simple_classification_dataset(num_schedules)

x = []
for each_ele in x_data:
    x.append(each_ele[2:])

x = torch.Tensor(x).reshape(-1,2)
y = torch.Tensor(y).reshape((-1,1))

print('Toy problem generated, and data cleaned')

x_data_test, y_test = create_simple_classification_dataset(10)

x_test = []
for each_ele in x_data_test:
    x_test.append(each_ele[2:])

x_test = torch.Tensor(x_test).reshape(-1,2)
y_test = torch.Tensor(y_test).reshape((-1,1))

print('test set generated')
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


input_size = 2  # Just the x dimension
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
    for i in range(10):
        chosen_schedule_start = int(schedule_starts[i])
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            optimizer.zero_grad()
            pred = MLP(x_test[each_t])
            loss = F.cross_entropy(pred.reshape(1, 2), y_test[each_t].long())
            acc = (pred.argmax(dim=-1) == y_test[each_t].item()).to(torch.float32).mean()
            test_losses.append(loss.item())
            test_accs.append(acc.mean().item())
    print('Loss: {}, Accuracy: {}'.format(np.mean(test_losses), np.mean(test_accs)))
print('Finished Training')


