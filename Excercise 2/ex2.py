import numpy as np
import matplotlib.pyplot as plt


def gradient():
    pass


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def generate_training_data(tags):
    training_set = []
    for tag in tags:
        fxy = np.random.normal(2 * tag, 1, 100)
        training_set.extend(fxy)
    print training_set


def main():
    tags = [1, 2, 3]
    generate_training_data(tags)


if __name__ == '__main__':
    main()
