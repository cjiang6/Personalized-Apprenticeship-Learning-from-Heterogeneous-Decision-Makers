"""
GMM to NN
created on May 19, 2019 by Ghost
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from base_testing_environment.toy_result_files_hetero.generate_environment import create_simple_classification_dataset
from base_testing_environment.utils.accuracy_measures import compute_specificity, compute_sensitivity
from sklearn import mixture
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)  # ensures repeatability
np.random.seed(50)


class Classifier_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Classifier_MLP, self).__init__()
        self.h1 = nn.Linear(in_dim+2, hidden_dim)
        self.h2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.log_softmax(self.out(x), dim=0)
        return x

def make_cluster_data(x, y):
    clust_x = []
    clust_y = []
    new_x = []
    new_y = []
    for i in range(len(x)):
        if i % 20 == 0 and i != 0:
            clust_x.append(new_x)
            clust_y.append(new_y)
            new_x = []
            new_y = []
        else:
            new_x.append(x[i])
            new_y.append(y[i])
    return clust_x, clust_y



def create_iterables():
    """
    adds all possible state combinations
    :return:
    """
    iterables = [[0, 1], [0, 1]]
    states = []
    for t in itertools.product(*iterables):
        states.append(t)

    return states


def create_binary_inputs(x, mean_input):
    total_binary_embeddings = np.zeros((0))
    for counter,j in enumerate(x):
        embedding_copy = np.zeros((1, 2))
        for i, each_element in enumerate(mean_input):
            if each_element > j[i].item():
                embedding_copy[0][i] = 0
            else:
                embedding_copy[0][i] = 1

        if counter == 0:
            total_binary_embeddings = embedding_copy
        else:
            total_binary_embeddings = np.vstack((total_binary_embeddings, embedding_copy))

    return total_binary_embeddings

def pass_in_embedding_out_state_ID(states, binary):
    """
    pass in a binary embedding, and itll return the state id
    :param binary:
    :return:
    """
    binary_as_tuple = tuple(binary)
    index = states.index(binary_as_tuple)
    return index



def make_matrices(binary_inputs, y, num_scheds, states):
    """
            creates matrices bases on the binary embeddings
            :return:
            """
    matrices = []
    schedule_starts = np.linspace(0, 20 * (num_scheds - 1), num=num_scheds)
    for i in range(num_schedules):
        m = np.zeros((4, 2))
        matrices.append(m)
    for i, each_matrix in enumerate(matrices):
        # lets look at elements of schedule 1
        chosen_schedule_start = int(schedule_starts[i])
        for j in range(chosen_schedule_start, chosen_schedule_start + 20):
            binary_embedding = binary_inputs[j]
            index = pass_in_embedding_out_state_ID(states, binary_embedding)
            # action taken at this instance
            action = y[j]
            each_matrix[index][int(action)] += 1
        total_sum = each_matrix.sum()
        matrices[i] = np.divide(each_matrix, total_sum)

    print('n matrices have been generated')
    return matrices

def make_cluster_matrices(x, y, num_scheds):
    mean_input = torch.mean(x,dim=0)
    states = create_iterables()
    binary_inputs = create_binary_inputs(x, mean_input)
    matrices = make_matrices(binary_inputs, y,num_scheds, states)
    return matrices, mean_input, states


def cluster_matrices(matrices, num_schedules):
    # vectorize each matrix
    vectorized_set = []
    for i in matrices:
        vectorized = i.reshape(2 * 4, 1)
        vectorized_set.append(vectorized)
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full', random_state=0) # random state makes it deterministic
    # Fitting the input data
    new_set = np.hstack(tuple(vectorized_set)).reshape(num_schedules, 8)
    gmm_model = gmm.fit(np.asarray(new_set))
    labels = gmm_model.predict_proba(np.asarray(new_set))
    return gmm_model, labels



num_schedules = 50
x_data, y = create_simple_classification_dataset(num_schedules, train=True)

x = []
for each_ele in x_data:
    x.append(each_ele[2:])

x = torch.Tensor(x).reshape(-1, 2)
y = torch.Tensor(y).reshape((-1, 1))

print('Toy problem generated, and data cleaned')





input_size = 2  # Just the x dimension
hidden_size = 10  # The number of nodes at the hidden layer
num_classes = 2  # The number of output classes. In this case, from 0 to 1
learning_rate = 1e-3  # The speed of convergence

MLP = Classifier_MLP(in_dim=input_size, hidden_dim=hidden_size, out_dim=num_classes)
optimizer = torch.optim.Adam(MLP.parameters(), lr=learning_rate)


# cluster data
# transfer data into a set of schedules



matrices, mean_input, states = make_cluster_matrices(x,y, num_schedules)
gmm_model , labels = cluster_matrices(matrices, num_schedules)

epochs = 100
schedule_starts = np.linspace(0, 20 * (num_schedules - 1), num=num_schedules)
for epoch in range(epochs):  # loop over the dataset multiple times
    # for batch, (x_train, y_train) in enumerate(train_loader):
    for i in range(num_schedules):
        chosen_schedule_start = int(np.random.choice(schedule_starts))
        schedule_num = int(chosen_schedule_start / 20)
        cluster_num = labels[schedule_num]
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            optimizer.zero_grad()
            input = torch.cat([x[each_t],torch.Tensor(cluster_num)])
            pred = MLP(input)
            loss = F.cross_entropy(pred.reshape(1, 2), y[each_t].long())
            loss.backward()
            optimizer.step()
    learning_rate /= 1.1

print('Finished Training')

### REAL TEST


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
    current_schedule_matrix = np.zeros((4, 2))
    chosen_schedule_start = int(schedule_starts[i])
    for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
        print(current_schedule_matrix)
        if current_schedule_matrix.sum() == 0:
            cluster_num = gmm_model.predict_proba(current_schedule_matrix.reshape(1,-1))
        else:
            matrix = np.divide(current_schedule_matrix, current_schedule_matrix.sum())
            cluster_num = gmm_model.predict_proba(matrix.reshape(1,-1))
        # print(cluster_num)
        optimizer.zero_grad()
        input = torch.cat([x[each_t], torch.Tensor(cluster_num).reshape(2)])
        pred = MLP(input)
        loss = F.cross_entropy(pred.reshape(1, 2), y_test[each_t].long())
        preds[i].append(pred.argmax(dim=-1).item())
        actual[i].append(y_test[each_t].item())
        acc = (pred.argmax(dim=-1) == y_test[each_t].item()).to(torch.float32).mean()
        test_losses.append(loss.item())
        test_accs.append(acc.mean().item())


        # update matrix

        embedding_copy = np.zeros((1, 2))
        for z, each_element in enumerate(mean_input):
            if each_element > x_test[each_t][z].item():
                embedding_copy[0][z] = 0
            else:
                embedding_copy[0][z] = 1
        index = pass_in_embedding_out_state_ID(states, embedding_copy[0])
        action = y_test[each_t]
        current_schedule_matrix[index][int(action)] += 1




    per_schedule_test_accs.append(np.mean(test_accs))

sensitivity, specificity = compute_sensitivity(preds, actual), compute_specificity(preds, actual)

print('Loss: {}, Accuracy: {}'.format(np.mean(test_losses), np.mean(test_accs)))
print('per sched accuracy: ', np.mean(per_schedule_test_accs))
print('mean sensitivity: ', sensitivity, ', mean specificity: ', specificity)
# Compute sensitivity and specificity (ideally these should be very high)
file = open('heterogeneous_toy_env_results.txt', 'a')
file.write('gmm -> NN: mean: ' +
           str(np.mean(per_schedule_test_accs)) +
           ', std: ' + str(np.std(per_schedule_test_accs)) +
            ', sensitivity: ' + str(sensitivity) + ', specificity: '+ str(specificity) +
           ', Distribution of Class: 0: ' + str(percent_of_zeros) + ', 1: ' + str(1 - percent_of_zeros) +
           '\n')
file.close()
