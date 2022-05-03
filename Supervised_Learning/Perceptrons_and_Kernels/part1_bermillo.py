import numpy as np
import pandas as pd


def prediction(y_hat):
    return 1 if y_hat >= 0 else -1


def accuracy(X, y, w):
    N = len(y)
    return (np.sum( y == np.array([prediction(w.T.dot(x)) for x in X]) ) / N) * 100


def avg_perceptron(train_data, test_data, max_iter=100, shuffle=True):
    # Initialize performance tracking variables and length of data
    N = len(train_data)
    train_w_acc, train_avg_w_acc = [], []
    test_w_acc, test_avg_w_acc = [], []

    # Split data into features and labels
    X_train, y_train = train_data.drop(columns=['Response']).to_numpy(), train_data['Response'].to_numpy()
    X_test, y_test = test_data.drop(columns=['Response']).to_numpy(), test_data['Response'].to_numpy()

    # Initialize weights and counter
    w, avg_w = np.zeros(X_train.shape[1]), np.zeros(X_train.shape[1])
    counter = 1

    # Run training
    for iteration in range(max_iter):

        if shuffle:
            # Shuffle the order of data
            shuffled_train = train_data.sample(frac=1)

            # Split shuffled train data to features and labels
            X_train = shuffled_train.drop(columns=['Response']).to_numpy()
            y_train = shuffled_train['Response'].to_numpy()

        # Iterate through each training example
        for i in range(N):
            # Prediction is incorrect
            if y_train[i] * (w.T.dot(X_train[i])) <= 0:
                # Weight update
                w += y_train[i] * X_train[i]

            # Compute average weights
            avg_w = (counter * avg_w + w) / (counter + 1)

            # Update counter
            counter += 1

        # Track training and validation accuracies for each iteration
        train_w_acc.append(accuracy(X_train, y_train, w))
        train_avg_w_acc.append(accuracy(X_train, y_train, avg_w))

        test_w_acc.append(accuracy(X_test, y_test, w))
        test_avg_w_acc.append(accuracy(X_test, y_test, avg_w))

        # report stats
        print('[Iteration {:3}] Training Accuracy: {:2.1f} | Validation Accuracy: {:2.1f}'.format(iteration,
                                                                                                  train_w_acc[-1],
                                                                                                  test_w_acc[-1]))

    return w, avg_w, {'train_w_acc': train_w_acc, 'train_avg_w_acc': train_avg_w_acc,
                      'test_w_acc': test_w_acc, 'test_avg_w_acc': test_avg_w_acc, }


if __name__ == '__main__':

    # Load data
    train_data = pd.concat([pd.read_csv('pa3_data/pa3_train_X.csv'), pd.read_csv('pa3_data/pa3_train_y.csv')], axis=1)
    test_data  = pd.concat([pd.read_csv('pa3_data/pa3_dev_X.csv'),   pd.read_csv('pa3_data/pa3_dev_y.csv')],   axis=1)

    # Run training
    w, avg_w, info = avg_perceptron(train_data, test_data, max_iter=100, shuffle=False)



