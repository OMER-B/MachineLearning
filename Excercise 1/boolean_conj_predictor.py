import numpy as np
import sys


def X_and_Y_from_file(file):
    '''
    Extracts the examples and set of tags from the file.
    :param file: given file of boolean strings
    :return: X set of examples and Y set of tags.
    '''
    training_examples = np.loadtxt(file, int)
    d = training_examples.shape[1] - 1
    X = training_examples[:, :d]
    Y = training_examples[:, d]
    return X, Y


def y_hat(atomic, negation, t):
    '''
    Calculates the prediction (y_hat)
    :param atomic: the atomic array of the hypothesis
    :param negation: the negation array of the hypothesis
    :param t: instance of example
    :return: true or false
    '''
    for i, x in enumerate(t):
        if atomic[i] == 1 and x == 1:
            return 0
        if negation[i] == 1 and x == 0:
            return 0
    return 1


def consistency_algorithm(examples, Y):
    '''
    Implements the consistency algorithm following the provided pseudo-code.
    :param examples: set of examples X
    :param Y: set of tags Y
    :return: the hypothesis in two arrays.
    '''
    atomic = [1] * examples.shape[1]
    negation = [1] * examples.shape[1]

    for t, instance in enumerate(examples):
        if Y[t] == 1 and y_hat(atomic, negation, instance) == 0:
            for index, x in enumerate(instance):
                if x == 0:
                    atomic[index] = 0
                if x == 1:
                    negation[index] = 0
    return atomic, negation


def save_output(atomic, negation):
    '''
    Saves the output.
    :param atomic: the atomic array of the hypothesis
    :param negation: the negation array of the hypothesis
    :return: none
    '''
    string = ""
    for i, a in enumerate(atomic):
        if atomic[i] == 1:
            string += 'x' + str(i + 1) + ','
        if negation[i] == 1:
            string += "not(x" + str(i + 1) + "),"
    string = string[:-1]

    with open('output.txt', 'w') as output:
        output.write(string)


def main():
    X, Y = X_and_Y_from_file(sys.argv[1])
    atomic, negation = consistency_algorithm(X, Y)
    save_output(atomic, negation)


if __name__ == '__main__':
    main()
