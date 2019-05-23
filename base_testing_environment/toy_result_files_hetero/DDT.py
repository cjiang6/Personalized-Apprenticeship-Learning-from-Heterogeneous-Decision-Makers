import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_testing_environment.toy_result_files_hetero.generate_environment import create_simple_classification_dataset
from base_testing_environment.utils.accuracy_measures import compute_specificity, compute_sensitivity
from base_testing_environment.prolonet import ProLoNet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)  # ensures repeatability
np.random.seed(50)

num_schedules = 50
x_data, y = create_simple_classification_dataset(num_schedules, train=True)

x = []
for each_ele in x_data:
    x.append(each_ele[2:])

x = torch.Tensor(x).reshape(-1, 1, 2)
y = torch.Tensor(y).reshape((-1, 1))

print('Toy problem generated, and data cleaned')

x_data_test, y_test = create_simple_classification_dataset(10, cv=True)

x_test = []
for each_ele in x_data_test:
    x_test.append(each_ele[2:])

x_test = torch.Tensor(x_test).reshape(-1, 1, 2)
y_test = torch.Tensor(y_test).reshape((-1, 1))

print('test set generated')


input_size = 2  # Just the x dimension
hidden_size = 10  # The number of nodes at the hidden layer
num_classes = 2  # The number of output classes. In this case, from 0 to 1
learning_rate = 1e-3  # The speed of convergence

ddt = ProLoNet(input_dim=input_size,
               output_dim=num_classes,
               weights=None,
               comparators=None,
               leaves=4,
               is_value=False,
               vectorized=True,
               selectors=None)
optimizer = torch.optim.Adam(ddt.parameters(), lr=learning_rate)

epochs = 10
schedule_starts = np.linspace(0, 20 * (num_schedules - 1), num=num_schedules)
for epoch in range(epochs):  # loop over the dataset multiple times
    # for batch, (x_train, y_train) in enumerate(train_loader):
    for i in range(num_schedules):
        chosen_schedule_start = int(np.random.choice(schedule_starts))
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            optimizer.zero_grad()
            pred = ddt(x[each_t])
            loss = F.cross_entropy(pred.reshape(1, 2), y[each_t].long())
            loss.backward()
            optimizer.step()
    learning_rate /= 1.1
    test_losses, test_accs = [], []
    # for i, (x_test, y_test) in enumerate(test_loader):
    for i in range(10):
        chosen_schedule_start = int(schedule_starts[i])
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            optimizer.zero_grad()
            pred = ddt(x_test[each_t])
            loss = F.cross_entropy(pred.reshape(1, 2), y_test[each_t].long())
            acc = (pred.argmax(dim=-1) == y_test[each_t].item()).to(torch.float32).mean()
            test_losses.append(loss.item())
            test_accs.append(acc.mean().item())
    print('Loss: {}, Accuracy: {}'.format(np.mean(test_losses), np.mean(test_accs)))
print('Finished Training')

### REAL TEST

x_data_test, y_test, percent_of_zeros = create_simple_classification_dataset(50, True)

x_test = []
for each_ele in x_data_test:
    x_test.append(each_ele[2:])

x_test = torch.Tensor(x_test).reshape(-1, 1, 2)
y_test = torch.Tensor(y_test).reshape((-1, 1))
test_losses, test_accs = [], []
per_schedule_test_losses, per_schedule_test_accs = [], []
preds, actual = [[] for _ in range(50)], [[] for _ in range(50)]
for i in range(50):
    chosen_schedule_start = int(schedule_starts[i])
    for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
        optimizer.zero_grad()
        pred = ddt(x_test[each_t])
        loss = F.cross_entropy(pred.reshape(1, 2), y_test[each_t].long())
        preds[i].append(pred.argmax(dim=-1).item())
        actual[i].append(y_test[each_t].item())
        acc = (pred.argmax(dim=-1) == y_test[each_t].item()).to(torch.float32).mean()
        test_losses.append(loss.item())
        test_accs.append(acc.mean().item())
    per_schedule_test_accs.append(np.mean(test_accs))

sensitivity, specificity = compute_sensitivity(preds, actual), compute_specificity(preds, actual)

print('Loss: {}, Accuracy: {}'.format(np.mean(test_losses), np.mean(test_accs)))
print('per sched accuracy: ', np.mean(per_schedule_test_accs))
print('mean sensitivity: ', sensitivity, ', mean specificity: ', specificity)
# Compute sensitivity and specificity (ideally these should be very high)
file = open('heterogeneous_toy_env_results.txt', 'a')
file.write('DDT: mean: ' +
           str(np.mean(per_schedule_test_accs)) +
           ', std: ' + str(np.std(per_schedule_test_accs)) +
            ', sensitivity: ' + str(sensitivity) + ', specificity: '+ str(specificity) +
           ', Distribution of Class: 0: ' + str(percent_of_zeros) + '1: ' + str(1 - percent_of_zeros) +
           '\n')
file.close()
