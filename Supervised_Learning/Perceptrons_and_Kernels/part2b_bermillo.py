import time
import numpy as np
import pandas as pd


def kernelized_predict(y, K, alpha):
    N = len(y)

    # Compute and return predictions
    return np.sign(K.dot(alpha * y))


def kernelized_accuracy(y, y_hat):
    # Return correct percentage
    return (np.sum(y == y_hat) / len(y)) * 100


def polynomial_kernel(X1, X2, p):
    return np.power(1 + X1.dot(X2.T), p)


def batch_polynomial_kernelized_perceptron(train_data, test_data, kernel, max_iter=100, p=1, lr=1):
    # Initialize performance tracking variables and length of data
    N = len(train_data)
    train_acc = []
    test_acc = []

    # Split data into features and labels
    X_train, y_train = train_data.drop(columns=['Response']).to_numpy(), train_data['Response'].to_numpy()
    X_test, y_test = test_data.drop(columns=['Response']).to_numpy(), test_data['Response'].to_numpy()

    # Initialize weights
    alpha = np.zeros(N)

    # Compute Gram Matrix
    K_train = kernel(X_train, X_train, p)
    K_test = kernel(X_test, X_train, p)

    # Track time to run each iteration
    elapsed_times = []

    # Run training
    for iteration in range(max_iter):

        # Time each iteration
        start = time.perf_counter()

        # Make prediction for x_i
        u = K_train.dot(alpha * y_train)

        # Case if the prediction is incorrect
        for i in range(N):
            if u[i] * y_train[i] <= 0:
                alpha[i] += lr

        # End iteration time
        elapsed_times.append(time.perf_counter() - start)

        # Track training and validation accuracies for each iteration
        train_acc.append(kernelized_accuracy(y_train, kernelized_predict(y_train, K_train, alpha)))
        test_acc.append(kernelized_accuracy(y_test, kernelized_predict(y_train, K_test, alpha)))

        # Report stats
        print(
            '[Iteration {:3}] Training Accuracy: {:2.1f} | Validation Accuracy: {:2.1f} | Iteration Runtime: {:5.2f}'.format(
                iteration,
                train_acc[-1],
                test_acc[-1],
                elapsed_times[-1]))

    return alpha, {'train_acc': train_acc, 'test_acc': test_acc, 'elapsed_times': elapsed_times}


if __name__ == '__main__':
    # Load data
    train_data = pd.concat([pd.read_csv('pa3_data/pa3_train_X.csv'), pd.read_csv('pa3_data/pa3_train_y.csv')], axis=1)
    test_data  = pd.concat([pd.read_csv('pa3_data/pa3_dev_X.csv'), pd.read_csv('pa3_data/pa3_dev_y.csv')],     axis=1)

    # Run training
    alpha, info = batch_polynomial_kernelized_perceptron(train_data, test_data, polynomial_kernel, max_iter=100, p=1, lr=1)

    # Report total runtime
    print('Batch Algorithm Total Runtime: {:.2f}s'.format(np.sum(info['elapsed_times'])))
