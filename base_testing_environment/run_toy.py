import torch
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
#from model import BNN_Gaussian
from base_testing_environment.my_bnn import BNN_Gaussian
from base_testing_environment.my_train import train
from base_testing_environment.using_bayes import create_simple_classification_dataset



def main():
    device = 'cpu'
    nb_samples = 2
    train_batch_size = 20
    test_batch_size = 1
    lr = 0.001
    beta_1 = 0.5
    beta_2 = 0.999
    nb_workers = 32
    nb_epochs = 100
    num_schedules = 5
    train_data, labels = create_simple_classification_dataset(num_schedules, homog=True)
    bnn = BNN_Gaussian(8, 0.1)
    bnn.to(device)
    train(bnn, train_data, labels, nb_samples, nb_epochs, train_batch_size, test_batch_size, num_schedules, lr, beta_1, beta_2, nb_workers, device)

if __name__ == "__main__":
    main()
