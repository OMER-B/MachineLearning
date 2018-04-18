import numpy as np
import matplotlib.pyplot as plt


def f(x, miu):
    """
    Calculate the probability function for x and miu.
    :param x: x to calculate for.
    :param miu: Mean of the distribution.
    :return: Probability density of the normal distribution.
    """
    return 1 / np.sqrt(2 * np.pi) * np.exp((-(x - miu) ** 2) / 2)


def softmax(x):
    """
    Calculate the softmax of x (= wx + b)
    :param x: Should be (w*x+b)
    :return: Softmax result of x
    """
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def generate_training_data(Classes, no_examples=100):
    """
    Generate training data for the classes.
    :param Classes: Array of classes.
    :param no_examples: Number of examples to generate for each class. The default is 100 examples.
    :return: Array of tuples of the form (x, y).
    """
    training_set = []
    y = []
    for tag in Classes:
        fxy = np.random.normal(2 * tag, 1, no_examples)  # f(x | y = a) = N(2a, 1), a = 1, 2, 3
        training_set.extend(fxy)
        y.extend([tag] * no_examples)
    return zip(training_set, y)


def train(training_set, w, b, epochs=200, etaw=0.01, etab=0.01):
    """
    Train the data set.
    :param training_set: Set to train.
    :param w: Weights to train by.
    :param b: Bias to train by.
    :param epochs: Number of epochs for the training. Default is 200.
    :param etaw: Learn rate of w. Default is 0.01.
    :param etab: Learn rate of b. Default is 0.01.
    :return: Vectors of trained weights and bias.
    """
    for i in range(epochs):
        np.random.shuffle(training_set)
        for x, y in training_set:
            sftmx = softmax(w * x + b)
            dw = x * sftmx
            dw[y - 1] = x * sftmx[y - 1] - x
            db = sftmx
            db[y - 1] = sftmx[y - 1] - 1
            w = w - etaw * dw  # w^t = w^(t-1) - eta*dw
            b = b - etab * db  # b^t = b^(t-1) - eta*db
    return w, b


def plot(classes, w, b, start=0, end=10, den=100):
    """
    Plots the probability density graph and the learned probability graph.
    :param classes: Classes to draw by.
    :param w: Weights.
    :param b: Bias.
    :param start: Start point of the graph. Default is from 0.
    :param end: End point of the graph. Default is to 10.
    :param den: Density to draw. Default is 100 points.
    :return:
    """
    density = np.linspace(start, end, den)  # Draw the graph for x in the range [0,10]  (+ density)
    true_distribution = []
    test = []
    for x in density:
        pdf = [f(x, 2 * c) for c in classes]
        true_distribution.append(pdf[0] / sum(pdf))
        test.append(softmax(w * x + b)[0])

    # Graph styling #
    plt.rcParams.update({'font.size': 9})
    plt.xlabel("x")
    plt.ylabel("Probability")
    plt.title("Posterior Probabilities")
    plt.box(False)
    plt.minorticks_on()
    plt.tick_params(direction='out', color='black')
    plt.grid(color='black', alpha=0.01, linewidth=0.3, which='both')
    plt.plot(density, true_distribution, color='y', linestyle='dotted', linewidth=1)
    plt.plot(density, test, color='purple', linewidth=2)
    plt.legend(('True', 'Estimated'),
               fancybox=False, edgecolor='white', fontsize='small')
    plt.show()


def main(classes=None):
    if classes is None:
        classes = [1, 2, 3]
    training_set = generate_training_data(classes)
    weights = np.random.random((len(classes), 1))
    bias = np.random.random((len(classes), 1))
    w, b = train(training_set, weights, bias)
    plot(classes, w, b)


if __name__ == '__main__':
    main()
