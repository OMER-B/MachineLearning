# coding=utf-8
import numpy as np

softmax = lambda x: np.exp(x - np.max(x)) / (np.exp(x - np.max(x))).sum()

sigmoid = lambda x: 1 / (1 + np.exp(-x))
d_sigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))

relu = lambda x: np.maximum(0, x)
d_relu = lambda x: 1 if x > 0 else 0

leaky_relu = lambda x: np.maximum(0.01, x)
d_leaky_relu = lambda x: 1 if x > 0 else 0

prelu = lambda a, x: np.maximum(a * x, x)
d_prelu = lambda a, x: 1 if x > 0 else a


def bprop(cache, y):  # this is good
    x, z1, h1, z2, h2 = [cache[key] for key in ('x', 'z1', 'h1', 'z2', 'h2')]
    y_z = np.zeros(10)
    y_z.shape = (h2.shape[0], 1)
    y_z[int(y)] = 1
    dz2 = (h2 - y_z)  # dL/dh2
    dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
    db2 = dz2  # dL/dz2 * dz2/db2
    dh1 = np.dot(cache['W2'].T, dz2)
    dz1 = dh1 * d_sigmoid(z1)  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'db1': db1, 'dW1': dW1, 'db2': db2, 'dW2': dW2}


def forward(x, params, active_function):  # this is good
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    x.shape = (W1.shape[1], 1)
    z1 = np.dot(W1, x) + b1
    h1 = active_function(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    # y_z = np.zeros(10)
    # y_z[int(y)] = 1
    # loss = (h2, y_z)
    ret = {'x': x, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
    for key in params:
        ret[key] = params[key]
    return ret


def weights_bias(H, input=784, output=10):  # this is good
    # W1 = np.random.rand(H, input)
    # b1 = np.random.rand(H, 1)
    # W2 = np.random.rand(output, H)
    # b2 = np.random.rand(output, 1)
    W1 = np.random.uniform(-0.08, 0.08, [H, input])  # [Hx784] weights
    b1 = np.random.uniform(-0.08, 0.08, [H, 1])  # [Hx1] bias
    W2 = np.random.uniform(-0.08, 0.08, [10, H])  # [10xH] weights
    b2 = np.random.uniform(-0.08, 0.08, [10, 1])  # [10x1] bias
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return params


def load(train_x="train_x", train_y="train_y", test_x="test_x", validation=0.2):  # this is good
    x = np.loadtxt(train_x) / 255.0
    y = np.loadtxt(train_y)
    test = np.loadtxt(test_x)
    size = int(x.shape[0] * validation)
    print size
    return x, y, x[-size:], y[-size:], test


def predict_on_dev(params, active_function, dev_x, dev_y):
    avg_loss = 0.0
    acc = 0.0
    sum_loss = good = 0.0
    for x, y in zip(dev_x, dev_y):
        out = forward(x, params, active_function)
        loss = Loss(out['h2'], y)
        sum_loss += loss
        if np.argmax(out['h2']) == y:
            good += 1
        acc = good / dev_x.shape[0]
        avg_loss = sum_loss / dev_x.shape[0]
    return avg_loss, acc


def Loss(out, y):
    loss = -np.log(out[int(y)])
    return loss


def shuffle(a, b):  # this is good
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)
    return a[indices], b[indices]


def update_weights_sgd(params, gradients, eta):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    dW1, db1, dW2, db2 = [gradients[key] for key in ('dW1', 'db1', 'dW2', 'db2')]
    W1 -= eta * dW1
    W2 -= eta * dW2
    b1 -= eta * db1
    b2 -= eta * db2
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def main(active_function=sigmoid, hidden_layer=100, eta=0.01, epochs=20, batch_size=1, validation=0.2):
    train_x, train_y, dev_x, dev_y, test_x = load(validation=validation)
    params = weights_bias(hidden_layer)
    # fprop_cache = fprop(train_x, train_y, params, active_function)
    # bprop_cache = bprop(fprop_cache)
    print "┌───────┬─────────────────┬─────────────────┬─────────────────┐"
    print "│ Epoch │ Avg. Train Loss │  Avg. dev loss  │ Acc on dev      │"
    for i in xrange(epochs):
        sum_loss = 0.0
        train_x, train_y = shuffle(train_x, train_y)
        for x, y in zip(train_x, train_y):
            out = forward(x, params, active_function)
            sum_loss += Loss(out['h2'], y)
            gradients = bprop(out, y)
            params = update_weights_sgd(params, gradients, eta)
        dev_loss, acc = predict_on_dev(params, active_function, dev_x, dev_y)
        print "│%6d │ %-15s │ %-15s │ %-16s│" % (
            i, str(sum_loss / train_y.shape[0]), str(dev_loss), str(acc * 100) + "%")
    with open("test.pred2", "w") as f:
        for x in (test_x):
            out = forward(x, params, active_function)
            y_hat = out['h2'].argmax()
            f.write(str(y_hat) + "\n")


if __name__ == '__main__':
    main()
