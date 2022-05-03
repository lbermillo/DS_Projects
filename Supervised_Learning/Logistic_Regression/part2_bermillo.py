import numpy as np
import pandas as pd


def scale_input(x, cols=None):
    def scale(arr):
        return arr / arr.max()

    for col in cols:
        x[col] = scale(x[col])

    return x


# NumPy friendly numerically stable sigmoid [https://stackoverflow.com/a/62860170]
def sigmoid(x):
    return np.piecewise(x, [x > 0], [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))])


def accuracy(X, y, w):
    N = len(y)
    return np.sum(y == np.round(sigmoid(X.dot(w)))) / N


def train(X, y, lr, lambda_reg, epochs=1e4):
    # initialize weights and N
    w = np.random.rand(X.shape[1])
    N = X.shape[0]

    # keep epoch count for termination
    epoch = 0

    # keep track of accuracy for early stopping
    best_acc = accuracy(X, y, w)
    best_w = np.copy(w)

    # start training
    while epoch < epochs:

        # update weights w/out L2 norm
        w += (lr / N) * X.T.dot(y - sigmoid(X.dot(w)))

        # use L1 regularization
        w[:-1] = np.sign(w[:-1]) * list(map(lambda x: max(x, 0), np.abs(w[:-1]) - lr * lambda_reg))

        # evaluate training accuracy
        if epoch % 100 == 0:
            curr_acc = accuracy(X, y, w)

            # early stopping
            if best_acc > curr_acc:
                break
            else:
                best_acc = curr_acc
                best_w = np.copy(w)

        # update epoch
        epoch += 1

    return best_w


if __name__ == '__main__':
    # read in data
    X_train, y_train = pd.read_csv('PA2v1_data/pa2_train_X.csv'), pd.read_csv('PA2v1_data/pa2_train_y.csv')
    X_test, y_test = pd.read_csv('PA2v1_data/pa2_dev_X.csv'), pd.read_csv('PA2v1_data/pa2_dev_y.csv')

    # scale inputs
    scale_input(X_train, cols=['Age', 'Annual_Premium', 'Vintage'])
    scale_input(X_test, cols=['Age', 'Annual_Premium', 'Vintage'])

    best_w, best_acc = None, float('-inf')
    training_acc, validation_acc = {}, {}

    # run tests for lambda values of 1e-i, i=[0, 1, 2, 3, 4, 5]
    for param in ['1e-{}'.format(i) for i in range(6)]:
        w = train(X_train.to_numpy(), y_train.to_numpy().flatten(), lr=1e-1, lambda_reg=float(param))

        # evaluate on training and validation dataset
        train_acc = accuracy(X_train.to_numpy(), y_train.to_numpy().flatten(), w)
        validate_acc = accuracy(X_test.to_numpy(), y_test.to_numpy().flatten(), w)

        # report stats
        print('Lambda: {:>3} | Training Accuracy: {:.5f} | Validation Accuracy: {:.5f} | W zero count: {}'.format(
            param,
            train_acc,
            validate_acc,
            len(w) - np.count_nonzero(w)))

        # append to respective lists
        training_acc[param] = train_acc * 100
        validation_acc[param] = validate_acc * 100

        # keep track of weights with the best performance
        if best_acc < validate_acc:
            best_w = w
            best_acc = validate_acc

    feature_coeff = [(feature, weight) for feature, weight in zip(X_train.columns, best_w)]
    feature_coeff.sort(key=(lambda x: np.abs(x[1])), reverse=True)

    for feature, coeff in feature_coeff[:5]:
        print('{:25}: {:10.5f}'.format(feature, coeff))

    print('Features with zero weights: {}'.format(len(best_w) - np.count_nonzero(best_w)))
