import numpy as np
import matplotlib.pyplot as plt


def phi(x, miu):
    return 1 / np.sqrt(2 * np.pi) * np.exp((-(x - miu) ** 2) / 2)


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def generate_training_data(tags):
    training_set = []
    y = []
    for tag in tags:
        fxy = np.random.normal(2 * tag, 1, 100)
        training_set.extend(fxy)
        y.extend([tag] * 100)
    return zip(y, training_set)


def train(set, weights, bias, learn_rate=0.1):
    pass


def plot(tags):
    density = np.linspace(0, 10, 100)  # Graph density

    true_distribution = []
    test = []
    for x in density:
        cdf = [phi(x, 2 * label) for label in tags]
        true_distribution.append(cdf[0] / sum(cdf))
        test.append(cdf[1] / sum(cdf))

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


def main():
    tags = [1, 2, 3]
    training_set = generate_training_data(tags)

    weights = np.zeros(3)
    bias = np.zeros(3)
    train(training_set, weights, bias)
    plot(tags)


if __name__ == '__main__':
    main()
