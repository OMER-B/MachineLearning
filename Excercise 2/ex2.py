import numpy as np
import matplotlib.pyplot as plt


def phi(x, miu):
    """
    Calculate the phi function for x and miu.
    :param x:
    :param miu:
    :return:
    """
    return 1 / np.sqrt(2 * np.pi) * np.exp((-(x - miu) ** 2) / 2)


def softmax(x):
    """
    Calculate the softmax of x
    :param x: wx+b
    :return: softmax of (x)
    """
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def generate_training_data(tags, no_examples=100):
    """
    Generate training data.
    :param tags: tags array
    :param no_examples: number of examples.
    :return: zip of training set and y
    """
    training_set = []
    y = []
    for tag in tags:
        fxy = np.random.normal(2 * tag, 1, no_examples)  # f(x | y = a) = N(2a, 1), a = 1, 2, 3
        training_set.extend(fxy)
        y.extend([tag] * no_examples)
    return zip(training_set, y)


def train(training_set, w, b, epochs=10, learn_rate=0.1):
    """
    Train the data.
    :param training_set: training set
    :param w: weights
    :param b: bias
    :param epochs: number of epochs
    :param learn_rate: learning rate
    :return: learned weights and bias
    """
    for i in range(epochs):
        np.random.shuffle(training_set)
        for x, y in training_set:
            sftmx = softmax(w * x + b)
            dw = x * sftmx
            dw[y - 1] = x * sftmx[y - 1] - x
            db = sftmx
            db[y - 1] = sftmx[y - 1] - 1
            w = w - learn_rate * dw  # w^t = w^(t-1) - eta*dw
            b = b - learn_rate * db  # b^t = b^(t-1) - eta*db
    return w, b


def plot(tags, w, b, start=0, end=10, den=100):
    """
    Plot the graphs
    :param tags:
    :param w:
    :param b:
    :param start:
    :param end:
    :param den:
    :return:
    """
    density = np.linspace(start, end, den)  # Draw the graph for x in the range [0,10]  (+ density)
    true_distribution = []
    test = []
    for x in density:
        cdf = [phi(x, 2 * tag) for tag in tags]
        true_distribution.append(cdf[0] / sum(cdf))
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
    plt.plot(density, true_distribution, color='y', linestyle='dashed', linewidth=1)
    plt.plot(density, test, color='purple', linewidth=2)
    plt.legend(('True', 'Estimated'),
               fancybox=False, edgecolor='white', fontsize='small')
    plt.show()


def main(tags=None):
    if tags is None:
        tags = [1, 2, 3]
    training_set = generate_training_data(tags)
    weights = np.random.random((len(tags), 1))
    bias = np.random.random((len(tags), 1))
    w, b = train(training_set, weights, bias)
    plot(tags, w, b)


if __name__ == '__main__':
    main()
