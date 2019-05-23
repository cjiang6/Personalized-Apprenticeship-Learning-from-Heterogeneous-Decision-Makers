"""
file to generate homogeneous dataset
created on May 17, 2019 by Ghost
"""
import numpy as np


def create_simple_classification_dataset(n):
    """
    Regression version:
    y = {
    z * x if assignment = 1
    (2-z) * x if assignment = 2
    Classification version:
    y = q*x

    q = {
    1 if z >= 0 and assignment = 1
    0 if z < 0 and assignment = 1
    1 if z < 0 and assignment = 2
    0 if z >= 0 and assignment = 2
    :param n:
    :return:
    """

    # sample z from 0 to 1
    lst = [[] for _ in range(n * 20)]  # each list is a timestep
    label = [[] for _ in range(n * 20)]  # each list is a timestep
    for i in range(n):
        lam = 0

        # lam hold throughout schedule
        for count in range(20):
            z = np.random.normal(0, 1)
            x = np.random.choice([0, 1], p=[.15, .85])
            q = None
            if lam == 1:  # assignment 1
                if z >= 0:
                    q = 1
                else:
                    q = 0
            if lam == 0:  # assignment 2
                if z < 0:
                    q = 1
                else:
                    q = 0

            y = q * x

            lst[i * 20 + count].extend([lam, q, z, x])
            label[i * 20 + count].append(y)

    return lst, label


# test
def main():
    """
    run to generate 5 schedules
    :return:
    """
    x, y = create_simple_classification_dataset(5)
    count = 0
    for i in y:
        if i[0] == 0:
            count += 1
    print('percent of zeros', count / len(y))  # confirm that the distribution isn't too skewed.


if __name__ == '__main__':
    main()
