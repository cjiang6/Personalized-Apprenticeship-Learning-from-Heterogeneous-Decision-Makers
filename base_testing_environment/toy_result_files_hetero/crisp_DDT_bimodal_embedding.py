"""
NN w/ Bayes rule updating for embedding implementation evaluating train and test performance on a homogeneous dataset
created on May 17, 2019 by Ghost
"""
from base_testing_environment.toy_result_files_hetero.generate_environment import create_simple_classification_dataset
from base_testing_environment.utils.accuracy_measures import compute_specificity, compute_sensitivity
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_testing_environment.prolonet import ProLoNet
from Ghost.tree_nets.utils.fuzzy_to_crispy import convert_to_crisp, convert_to_complicated_crisp

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(50)  # ensures repeatability
np.random.seed(50)


def get_embedding_given_dist(dist):
    choice = np.random.choice([0, 1], p=[dist[0], dist[1]])
    if choice == 0:
        return torch.Tensor([1, 0]), 0
    else:
        return torch.Tensor([0, 1]), 1


def get_most_likely_embedding_given_dist(dist):
    choice = np.argmax(dist)
    if choice == 0:
        return torch.Tensor([1, 0])
    else:
        return torch.Tensor([0, 1])


def train():
    # Training set generation
    num_schedules = 50
    x_data, y = create_simple_classification_dataset(num_schedules, train=True)

    x = []
    for each_ele in x_data:
        x.append(each_ele[2:])

    x = torch.Tensor(x).reshape(-1, 1, 2)
    y = torch.Tensor(y).reshape((-1, 1))

    print('Toy problem generated, and data cleaned')

    input_size = 2  # Just the x and z dimension
    hidden_size = 10  # The number of nodes at the hidden layer
    num_classes = 2  # The number of output classes. In this case, from 0 to 1
    learning_rate = 1e-3  # The speed of convergence

    ddt = ProLoNet(input_dim=input_size,
                   output_dim=num_classes,
                   weights=None,
                   comparators=None,
                   leaves=32,
                   is_value=False,
                   bayesian_embedding_dim=2,
                   vectorized=True,
                   selectors=None)
    distributions = [np.ones(2) * 1 / 2 for _ in range(num_schedules)]
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(ddt.parameters(), lr=.001)
    schedule_starts = np.linspace(0, int(num_schedules * 20 - 20), num=num_schedules)
    epochs = 150

    for epoch in range(epochs):  # loop over the dataset multiple times
        for _ in range(num_schedules):
            # choose a schedule
            chosen_schedule_start = int(np.random.choice(schedule_starts))
            schedule_num = int(chosen_schedule_start / 20)
            # embedding_given_dis = get_embedding_given_dist(distributions[schedule_num])
            prod = [0, 0]
            # for each embedding
            opt.zero_grad()
            test_models = [0, 0]
            losses = [0, 0]
            for count, each_i in enumerate([torch.Tensor([1, 0]), torch.Tensor([0, 1])]):
                # model_copy = NeuronBNN()
                # model_copy.load_state_dict(model.state_dict())
                # test_models[count] = model_copy
                ddt.set_bayesian_embedding(each_i)
                tally = 1
                avg_loss_over_schedule = 0

                for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
                    x_t = x[each_t]
                    output = ddt.forward(x_t).reshape(1, 2)
                    label = y[each_t]
                    label = torch.Tensor([label]).reshape(1)
                    label = label.long()

                    loss = criterion(output, label)
                    # loss.backward()
                    losses[count] += loss
                    avg_loss_over_schedule += loss.item()

                    # print('output is ', torch.argmax(output).item(), ' label is ', label.item())
                    tally *= output[0][int(label.item())].item()
                prod[count] = tally * distributions[schedule_num][count]
                if max(distributions[schedule_num]) == distributions[schedule_num][count]:  # if you choose correctly
                    print("avg loss over schedule is: ", avg_loss_over_schedule / 20)

            normalization_factor = sum(prod)
            prod = [i / normalization_factor for i in prod]

            # update each
            distributions[schedule_num][0] = prod[0]
            distributions[schedule_num][1] = prod[1]

            normalization_factor_for_dist = sum(distributions[schedule_num])
            distributions[schedule_num] /= normalization_factor_for_dist  # [i/normalization_factor_for_dist for i in distributions[schedule_num]]
            coin_flip = np.random.choice([0, 1], p=[distributions[schedule_num][0], distributions[schedule_num][1]])
            chosen_loss = losses[int(coin_flip)]
            chosen_loss.backward()
            # opt = torch.optim.SGD(test_models[coin_flip].parameters(), lr=.001)
            opt.step()
            # model.load_state_dict(test_models[coin_flip].state_dict())

    print('Finished Training')
    return ddt


# REAL TEST
def test(ddt):


    x_data_test, y_test, percent_of_zeros = create_simple_classification_dataset(50, get_percent_of_zeros=True)
    schedule_starts = np.linspace(0, int(50 * 20 - 20), num=50)
    x_test = []

    for each_ele in x_data_test:
        x_test.append(each_ele[2:])

    x_test = torch.Tensor(x_test).reshape(-1, 1, 2)
    y_test = torch.Tensor(y_test).reshape((-1, 1))

    test_losses, test_accs = [], []
    per_schedule_test_losses, per_schedule_test_accs = [], []
    preds, actual = [[] for _ in range(50)], [[] for _ in range(50)]
    test_distributions = [np.ones(2) * 1 / 2 for _ in range(50)]
    total_acc = []
    for i in range(50):
        chosen_schedule_start = int(schedule_starts[i])
        schedule_num = int(chosen_schedule_start / 20)
        embedding_given_dis, count = get_embedding_given_dist(test_distributions[schedule_num])
        prod = [.5, .5]
        acc = 0
        ddt.set_bayesian_embedding(embedding_given_dis)

        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            # at each timestep you what to resample the embedding

            x_t = x_test[each_t]
            output = ddt.forward(x_t).reshape(1, 2)

            label = y_test[each_t]
            label = torch.Tensor([label]).reshape(1)
            label = label.long()
            print('output is ', torch.argmax(output).item(), ' label is ', label.item())
            if torch.argmax(output).item() == label.item():
                acc += 1
            tally = output[0][int(label.item())].item()
            second_tally = output[0][int(not label.item())].item()
            prod[count] = tally * test_distributions[i][count]
            prod[int(not count)] *= second_tally * test_distributions[i][int(not count)]
            preds[i].append(torch.argmax(output).item())
            actual[i].append(label.item())
            normalization_factor = sum(prod)
            prod = [k / normalization_factor for k in prod]

            test_distributions[schedule_num][0] = prod[0]
            test_distributions[schedule_num][1] = prod[1]
            normalization_factor_for_dist = sum(test_distributions[schedule_num])
            test_distributions[schedule_num] /= normalization_factor_for_dist  # [i/normalization_factor_for_dist for i in distributions[schedule_num]]
            print('distribution at time ', each_t, ' is', test_distributions[schedule_num])
            if each_t % 20 < 5:
                embedding_given_dis, count = get_embedding_given_dist(test_distributions[schedule_num])
            else:
                embedding_given_dis = get_most_likely_embedding_given_dist(test_distributions[schedule_num])
            ddt.set_bayesian_embedding(embedding_given_dis)

        per_schedule_test_accs.append(acc / 20)
    # print('Loss: {}, Accuracy: {}'.format(0, np.mean(per_schedule_test_accs)))
    print(test_distributions)
    print('per sched accuracy: ', np.mean(per_schedule_test_accs))
    sensitivity, specificity = compute_sensitivity(preds, actual), compute_specificity(preds, actual)



    test_losses, test_accs = [], []
    per_schedule_test_losses, per_schedule_test_accs = [], []
    preds, actual = [[] for _ in range(50)], [[] for _ in range(50)]
    total_acc = []
    for i in range(50):
        chosen_schedule_start = int(schedule_starts[i])
        schedule_num = int(chosen_schedule_start / 20)
        embedding_given_dis, count = get_embedding_given_dist(test_distributions[schedule_num])
        prod = [.5, .5]
        acc = 0
        ddt.set_bayesian_embedding(embedding_given_dis)

        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            # at each timestep you what to resample the embedding

            x_t = x_test[each_t]
            output = ddt.forward(x_t).reshape(1, 2)

            label = y_test[each_t]
            label = torch.Tensor([label]).reshape(1)
            label = label.long()
            # print('output is ', torch.argmax(output).item(), ' label is ', label.item())
            if torch.argmax(output).item() == label.item():
                acc += 1
            tally = output[0][int(label.item())].item()
            second_tally = output[0][int(not label.item())].item()
            prod[count] = tally * test_distributions[i][count]
            prod[int(not count)] *= second_tally * test_distributions[i][int(not count)]
            preds[i].append(torch.argmax(output).item())
            actual[i].append(label.item())


        per_schedule_test_accs.append(acc / 20)
    # print('Loss: {}, Accuracy: {}'.format(0, np.mean(per_schedule_test_accs)))
    print('per sched accuracy: ', np.mean(per_schedule_test_accs))
    fuzzy_sensitivity, fuzzy_specificity = compute_sensitivity(preds, actual), compute_specificity(preds, actual)
    fuzzy_accuracy = np.mean(per_schedule_test_accs)

    ddt = convert_to_crisp(ddt, None)


    test_losses, test_accs = [], []
    per_schedule_test_losses, per_schedule_test_accs = [], []
    preds, actual = [[] for _ in range(50)], [[] for _ in range(50)]

    total_acc = []
    for i in range(50):
        chosen_schedule_start = int(schedule_starts[i])
        schedule_num = int(chosen_schedule_start / 20)
        embedding_given_dis, count = get_embedding_given_dist(test_distributions[schedule_num])
        prod = [.5, .5]
        acc = 0
        ddt.set_bayesian_embedding(embedding_given_dis)

        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            # at each timestep you what to resample the embedding

            x_t = x_test[each_t]
            output = ddt.forward(x_t).reshape(1, 2)

            label = y_test[each_t]
            label = torch.Tensor([label]).reshape(1)
            label = label.long()
            # print('output is ', torch.argmax(output).item(), ' label is ', label.item())
            if torch.argmax(output).item() == label.item():
                acc += 1
            tally = output[0][int(label.item())].item()
            second_tally = output[0][int(not label.item())].item()
            prod[count] = tally * test_distributions[i][count]
            prod[int(not count)] *= second_tally * test_distributions[i][int(not count)]
            preds[i].append(torch.argmax(output).item())
            actual[i].append(label.item())


        per_schedule_test_accs.append(acc / 20)
    # print('Loss: {}, Accuracy: {}'.format(0, np.mean(per_schedule_test_accs)))
    print('per sched accuracy: ', np.mean(per_schedule_test_accs))
    crisp_sensitivity, crisp_specificity = compute_sensitivity(preds, actual), compute_specificity(preds, actual)
    crisp_accuracy = np.mean(per_schedule_test_accs)













    print('mean crisp sensitivity: ', crisp_sensitivity, ', mean crisp specificity: ', crisp_specificity)
    file = open('heterogeneous_toy_env_results.txt', 'a')
    file.write('crisp DDT w/ bimodal embedding: crisp mean: ' +
               str(crisp_accuracy) +
               ', fuzzy mean: ' +
               str(fuzzy_accuracy) +
               ', crisp sensitivity: ' + str(crisp_sensitivity) + ', crisp specificity: ' + str(crisp_specificity) +
               ', fuzzy sensitivity: ' + str(fuzzy_sensitivity) + ', fuzzy specificity: ' + str(fuzzy_specificity) +
               ', Distribution of Class: 0: ' + str(percent_of_zeros) + ', 1: ' + str(1 - percent_of_zeros) +
               '\n')
    file.close()



def main():
    bnn = train()
    test(bnn)

if __name__ == '__main__':
    main()
