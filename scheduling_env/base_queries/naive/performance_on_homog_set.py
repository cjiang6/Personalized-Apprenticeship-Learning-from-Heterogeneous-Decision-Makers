"""
File to test performance of ProLoNet on homogeneous dataset
"""

import torch
import sys

# sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')
from scheduling_env.alpha_div import AlphaLoss
from Ghost.tree_nets.vectorized_prolonet import ProLoNet
import numpy as np
from scheduling_env.argument_parser import Logger
import pickle
from torch.autograd import Variable

from utils.naive_utils import create_new_dataset

sys.path.insert(0, '../')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)


# noinspection PyTypeChecker
class BDTTrain:
    """
    class structure to train the BDT with a certain alpha.
    This class handles training the BDT, evaluating the BDT, and saving
    """

    def __init__(self):
        self.arguments = Logger()
        self.alpha = .9
        self.num_schedules = 150
        self.home_dir = self.arguments.home_dir
        self.total_loss_array = []

        load_directory = '/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env/datasets/' + str(
            self.num_schedules) + '_inf_hetero_deadline_naive.pkl'


        self.data = pickle.load(open(load_directory, "rb"))
        self.X, self.Y, self.schedule_array = create_new_dataset(self.data, num_schedules=self.num_schedules)
        for i, each_element in enumerate(self.X):
            self.X[i] = each_element + list(range(20))


        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = ProLoNet(input_dim=len(self.X[0]),
                              weights=None,
                              comparators=None,
                              leaves=256,
                              output_dim=20,
                              bayesian_embedding_dim=None,
                              alpha=1.5,
                              use_gpu=True,
                              vectorized=False,
                              is_value=False)

        use_gpu = True
        if use_gpu:
            self.model = self.model.cuda()
        print(self.model.state_dict())
        params = list(self.model.parameters())
        self.opt = torch.optim.RMSprop([{'params': params}])
        self.num_iterations_predict_task = 0
        self.total_iterations = 0
        self.covergence_epsilon = .01
        self.when_to_save = 1000
        self.distribution_epsilon =  .0001




    def train(self):
        """
        Trains BDT.
        Randomly samples a schedule and timestep within that schedule, and passes in the corresponding data in an attempt to classify which task was scheduled
        :return:
        """


        criterion = torch.nn.NLLLoss()
        training_done = False
        cv_cutoff = .8 * len(self.X)

        while not training_done:
            # sample a timestep before the cutoff for cross_validation
            rand_timestep_within_sched = np.random.randint(cv_cutoff)
            input_nn = self.X[rand_timestep_within_sched]
            truth_nn = self.Y[rand_timestep_within_sched]


            # iterate over pairwise comparisons
            if torch.cuda.is_available():
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)).cuda()) # change to 5 to increase batch size
                truth_nn = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)).cuda().long())
            else:
                input_nn = Variable(torch.Tensor(np.asarray(input_nn).reshape(1, 242)))
                truth_nn = Variable(torch.Tensor(np.asarray(truth_nn).reshape(1)))

            self.opt.zero_grad()
            output_nn = self.model.forward(input_nn)

            loss_nn = criterion(output_nn, truth_nn)

            loss_nn.backward()
            self.opt.step()
            self.total_loss_array.append(loss_nn.item())

            total_iterations = len(self.total_loss_array)

            if total_iterations % 50 == 49:
                print('loss at', total_iterations, 'is', loss_nn.item())



            if total_iterations > 15000:
                training_done = True





    def evaluate(self, load_in_model=False):

        """
        Evaluate performance of a trained network.
        This is tested on 20% of the data and will be stored in a text file.
        :return:
        """
        percentage_accuracy_top1 = []

        for i, schedule in enumerate(self.schedule_array):
            if i < .8 * len(self.schedule_array):
                continue

            prediction_accuracy = 0
            for count in range(schedule[0], schedule[1] + 1):


                net_input = self.X[count]
                truth = self.Y[count]


                if torch.cuda.is_available():
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)).cuda())
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)).cuda().long())
                else:
                    input_nn = Variable(torch.Tensor(np.asarray(net_input).reshape(1, 242)))
                    truth = Variable(torch.Tensor(np.asarray(truth).reshape(1)))



                #####forward#####
                output = self.model.forward(input_nn)

                index = torch.argmax(output).item()






                if index == truth.item():
                        prediction_accuracy += 1

            print('Prediction Accuracy: top1: ', prediction_accuracy / 20)

            print('schedule num:', i)
            percentage_accuracy_top1.append(prediction_accuracy / 20)

        print(np.mean(percentage_accuracy_top1))





def main():
    """
    entry point for file
    :return:
    """


    trainer = BDTTrain()
    trainer.train()
    trainer.evaluate()



if __name__ == '__main__':
    main()
