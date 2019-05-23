import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable

def train(bnn, train_data, labels, nb_samples,
          nb_epochs=20, train_batch_size=20,
          test_batch_size=1, num_schedules=50,
          lr=0.001, beta_1=0.9, beta_2=0.999,
          nb_workers=4, device="cuda"):

    schedule_starts = np.linspace(0, 20 * (num_schedules - 1), num=num_schedules)
    # not_first_time = False
    # distributions = [np.array([.15, .1], dtype=float) for _ in range(num_schedules)]  # each one is mean, sigma
    optimizer = torch.optim.SGD(bnn.parameters(), lr / 10)
    phase = 'train'
    for epoch in range(nb_epochs):

        nb_correct = 0
        total_nll_loss = 0.
        total_kl_loss = 0.
        for _ in range(num_schedules):
            # x_data = []
            # y_data = []
            chosen_schedule_start = int(np.random.choice(schedule_starts))
            schedule_num = int(chosen_schedule_start / 20)
            # if phase == "train":
            #     nb_total = len(train_data) used for accuracy
            # else:
            #     nb_total = len(test_data)
            for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):

                x = train_data[each_t][2:]
                # x_data.append(x)
                # noinspection PyArgumentList
                x = torch.Tensor([x]).reshape((2))

                label = labels[each_t]
                # y_data.append(label)
                # noinspection PyArgumentList
                label = torch.Tensor([label]).reshape(1)
                label = Variable(label).long()

                x = torch.Tensor(x).to(device)
                y_true = label
                x = x.view(-1, 2)
                if phase == 'train':
                    optimizer.zero_grad()
                    x.requires_grad = True
                    y_true.requires_grad = False

                qw, pw, mle = bnn.forward_samples(x, y_true, nb_samples)
                kl_loss = (qw - pw) / 1 / len(y_true)
                total_kl_loss += kl_loss.item()
                nll_loss = -mle / len(y_true)
                total_nll_loss += nll_loss.item()
                loss = nll_loss + kl_loss
                output = bnn.forward(x, test=True)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                print('x is: ', x, ' output is: ', output, 'label is: ', label)
                y_pred = output.argmax(1)
                if torch.argmax(output).item() == label.item():
                    nb_correct += 1 # TODO: fix bug here
        # print(bnn.state_dict())
        print('{} Epoch: {}, NLL Loss: {:.3e}, KL loss:{:.3e}, Acc:{:.2f}%'.format(
            phase, epoch + 1, total_nll_loss / (20* num_schedules), total_kl_loss / (20* num_schedules), 100 * nb_correct / (20* num_schedules)
        ))

    # test
    print('TESTING LOOP STARTING')
    acc = 0
    for i in range(num_schedules):
        chosen_schedule_start = int(schedule_starts[i])
        schedule_num = int(chosen_schedule_start / 20)
        x_data = []
        y_data = []
        for each_t in range(chosen_schedule_start, chosen_schedule_start + 20):
            # at each timestep you what to resample the embedding

            x = train_data[each_t][2:]
            # x_data.append(x)
            x = torch.Tensor([x]).reshape((2))
            x = torch.Tensor(x).to(device)
            x = x.view(-1, 2)
            x.requires_grad = True
            label = labels[each_t]
            label = torch.Tensor([label]).reshape(1)
            label = Variable(label).long()
            y_true = label
            y_true.requires_grad = False
            qw, pw, mle = bnn.forward_samples(x, y_true, nb_samples)

            # output = predict(x)


            # y_data.append(label)

        # x = torch.Tensor(x_data).to(device)
            output = bnn.forward(x, test=True)
            print('x is: ', x, ' output is: ', output, 'label is: ', label)
        # for i,each_output in enumerate(output):
            if torch.argmax(output).item() == labels[each_t][0]:
                acc += 1


    print("accuracy: %d %%" % (100 * acc / (20*num_schedules)))
