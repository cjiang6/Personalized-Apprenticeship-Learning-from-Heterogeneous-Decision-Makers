"""
DT w/ Bayes rule updating for embedding implementation evaluating train and test performance on a heterogenous dataset
created on May 17, 2019 by Ghost
"""
from base_testing_environment.toy_result_files_hetero.generate_environment import create_simple_classification_dataset
from base_testing_environment.utils.accuracy_measures import compute_specificity, compute_sensitivity
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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


def DTtrain():
    """
    Pr[omega_i | game_g] = Pr[omega = omega_i] * \Prod_{data points j in game g} Pr[y_j | omega_i , x_j]

    Pr[assignment_g^{t+1} = i] = Pr[assignment_g^t = i] * Pr[omega_i | game g]
    :param num_schedules:
    :param data:
    :param labels:
    :return:
    """

    # Training set generation
    num_schedules = 50
    x_data, y = create_simple_classification_dataset(num_schedules, train=True)

    x = []
    for each_ele in x_data:
        x.append(each_ele[2:])

    x = torch.Tensor(x).reshape(-1, 2)
    y = torch.Tensor(y).reshape((-1, 1))

    print('Toy problem generated, and data cleaned')


    distributions = [np.ones(2) * 1 / 2 for _ in range(num_schedules)]
    epochs = 200
    schedule_starts = np.linspace(0, int(num_schedules * 20 - 20), num=num_schedules)
    clf = None
    for epoch in range(epochs):
        # data augmentation
        augmented_data = []
        answers = []
        which_embedding_was_chosen = []
        for j in range(num_schedules):
            # choose a schedule
            chosen_schedule_start = int(schedule_starts[j])
            schedule_num = int(chosen_schedule_start / 20)
            embedding_given_dis, count = get_embedding_given_dist(distributions[schedule_num])

            for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
                x_t = x[each_t]
                x_t = list(np.array(torch.cat([x_t, embedding_given_dis])))
                augmented_data.append(x_t)
                answers.append(y[each_t])
                which_embedding_was_chosen.append(count)
        clf = DecisionTreeClassifier(max_depth=4)
        clf.fit(augmented_data, answers)

        for j in range(num_schedules):
            # choose a schedule
            chosen_schedule_start = int(schedule_starts[j])
            schedule_num = int(chosen_schedule_start / 20)
            prod = [0, 0]
            # for each embedding
            count = which_embedding_was_chosen[chosen_schedule_start]
            tally = [1, 1]
            for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
                data_for_timestep = augmented_data[each_t]
                label_for_timestep = answers[each_t]

                y_pred = clf.predict_proba(np.array(data_for_timestep).reshape(1,-1))

                print('output is ', y_pred[0], ' label is ', label_for_timestep)
                label_for_timestep = int(label_for_timestep.item())
                tally[count] *= y_pred[0][label_for_timestep]
                tally[int(not count)] *= y_pred[0][int(not label_for_timestep)]

            prod[count] = tally[count] * distributions[schedule_num][count]
            prod[int(not count)] = tally[int(not count)] * distributions[schedule_num][int(not count)]

            normalization_factor = sum(prod)
            prod = [i / normalization_factor for i in prod]

            # update each
            distributions[schedule_num][0] = prod[0]
            distributions[schedule_num][1] = prod[1]

            normalization_factor_for_dist = sum(distributions[schedule_num])
            distributions[schedule_num] /= normalization_factor_for_dist  # [i/normalization_factor_for_dist for i in distributions[schedule_num]]


    print(distributions)
    print('finite')
    return clf

# REAL TEST

def DTtest(tree):
    """
    Pr[omega_i | game_g] = Pr[omega = omega_i] * \Prod_{data points j in game g} Pr[y_j | omega_i , x_j]
    Pr[assignment_g^{t+1} = i] = Pr[assignment_g^t = i] * Pr[omega_i | game g]
    :param tree:
    :param num_schedules:
    :param data:
    :param labels:
    :return:
    """
    num_schedules = 50
    x_data_test, y_test, percent_of_zeros = create_simple_classification_dataset(50, get_percent_of_zeros=True)
    schedule_starts = np.linspace(0, int(50 * 20 - 20), num=50)
    x_test = []
    preds, actual = [[] for _ in range(50)], [[] for _ in range(50)]
    for each_ele in x_data_test:
        x_test.append(each_ele[2:])

    data = torch.Tensor(x_test).reshape(-1, 2)
    y_test = torch.Tensor(y_test).reshape((-1, 1))
    distributions = [np.ones(2) * 1 / 2 for _ in range(50)]
    total_acc = []
    for i in range(num_schedules):
        # choose a schedule
        chosen_schedule_start = int(schedule_starts[i])
        schedule_num = int(chosen_schedule_start / 20)
        embedding_given_dis, count = get_embedding_given_dist(distributions[schedule_num])
        prod = [.5, .5]

        acc = 0
        tally = [1,1]
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            # at each timestep you what to resample the embedding
            if each_t == 92:
                print('hi')
            x = data[each_t]

            x = list(np.array(torch.cat([x, embedding_given_dis])))
            y_pred = tree.predict_proba(np.array(x).reshape(1,-1))

            label = y_test[each_t]

            print('output is ', y_pred[0], ' label is ', label)
            if np.argmax(y_pred[0]) == label:
                acc += 1
            preds[i].append(np.argmax(y_pred[0]))
            actual[i].append(label.item())
            tally[count] *= y_pred[0][int(label[0])]
            tally[int(not count)] *= y_pred[0][int(not label[0])]
            prod[count] = tally[count] * distributions[schedule_num][count]
            prod[int(not count)] = tally[int(not count)] * distributions[schedule_num][int(not count)]

            normalization_factor = sum(prod)
            prod = [k / normalization_factor for k in prod]

            distributions[schedule_num][0] = prod[0]
            distributions[schedule_num][1] = prod[1]
            normalization_factor_for_dist = sum(distributions[schedule_num])
            distributions[schedule_num] /= normalization_factor_for_dist  # [i/normalization_factor_for_dist for i in distributions[schedule_num]]

            print('distribution at time ', each_t, ' is', distributions[schedule_num])
            if each_t % 20 < 5:
                embedding_given_dis, count = get_embedding_given_dist(distributions[schedule_num])
            else:
                embedding_given_dis = get_most_likely_embedding_given_dist(distributions[schedule_num])
                count = np.argmax(distributions[schedule_num])

        total_acc.append(acc / 20)
    print('mean is ', np.mean(total_acc))
    print('finite')


    sensitivity, specificity = compute_sensitivity(preds, actual), compute_specificity(preds, actual)
    print('per sched accuracy: ', np.mean(total_acc))
    print('mean sensitivity: ', sensitivity, ', mean specificity: ', specificity)
    file = open('heterogeneous_toy_env_results.txt', 'a')
    file.write('DT w/ bimodal embedding: mean: ' +
               str(np.mean(total_acc)) +
               ', std: ' + str(np.std(total_acc)) +
               ', sensitivity: ' + str(sensitivity) + ', specificity: ' + str(specificity) +
               ', Distribution of Class: 0: ' + str(percent_of_zeros) + ', 1: ' + str(1 - percent_of_zeros) +
               '\n')
    file.close()



def main():
    tree = DTtrain()
    DTtest(tree)

if __name__ == '__main__':
    main()
