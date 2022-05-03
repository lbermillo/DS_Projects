import numpy as np
import matplotlib.pyplot as plt

from part0_bermillo import preprocess_data


def predict(x, w):
    return w.T.dot(x)


def mse(X, Y, w, N):
    return np.sum([(w.T.dot(X[i]) - Y[i]) ** 2 for i in range(N)]) / N


def train(X, Y, lr, epsilon=float('-inf'), epochs=float('inf')):
    N = len(Y)
    epoch = 0
    losses = []
    delta_L = float('inf')

    # 1. Start from some initial guess w0
    w = np.random.rand(X.shape[1])

    while np.linalg.norm(delta_L) > epsilon and epoch < epochs:
        # 2. Find the direction of steepest descent – opposite of the gradient direction 𝛻𝑓(𝐰)
        delta_L = (2 / N) * np.sum([(predict(X[i], w) - Y[i]) * X[i] for i in range(N)])

        # 3. Take a step opposite to that direction, i.e., along − 𝛻𝑓(𝐰) (step size controlled by 𝜆):
        # w_t+1 = w_t − 𝜆𝛻𝑓(w_t)
        w -= lr * delta_L

        # calculate loss
        loss = mse(X, Y, w, N)

        # append loss to array for plotting
        losses.append(loss)

        epoch += 1

    return w, losses


def plot_losses(losses, lr):
    plt.figure(figsize=(12, 10))
    plt.plot(losses[lr], label=str(lr))
    plt.title('Training Losses: LR={:.1e}'.format(lr))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # for preprocessing
    norm_cols = ['month', 'day', 'year', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
                 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
                 'sqft_living15', 'sqft_lot15']
    one_hot_cols = ['zipcode']

    X_train, Y_train = preprocess_data("PA1_train.csv", normalize=True, one_hot_cols=one_hot_cols, norm_cols=norm_cols)

    # dictionaries to track coefficients and losses for each learning rate
    coeffs, losses = {}, {}

    print('Part 1: Gradient Descent implementation\n')
    print('{:^20} | {:^20} | {:^20}'.format('Learning Rate', 'Loss', 'Epochs'))

    # run training for each learning rate
    for lr in [float('1e-{}'.format(i)) for i in range(6)]:
        train_coeff, train_losses = train(X_train.to_numpy(), Y_train.to_numpy(), lr=lr, epsilon=1e-3)

        # add training losses and coefficients to dictionary with their corresponding learning rate
        coeffs[lr], losses[lr] = train_coeff, train_losses

        print('{:^20} | {:^20.3f} | {:^20}'.format(lr, train_losses[-1], len(train_losses)))

    print('\n\n')

    # load and process test data
    X_test, Y_test = preprocess_data("PA1_dev.csv", one_hot_cols=one_hot_cols, norm_cols=norm_cols)

    print('Part 1b: Run MSE on training and validation data\n')
    print('{:^20} | {:^20} | {:^20}'.format('Learning Rate', 'Training Loss', 'Validation Loss'))

    # track best lr and mse for 1c
    best_lr, best_mse = 0, float('inf')

    for lr, coeff in coeffs.items():
        if lr < 1e-1:
            train_mse, validation_mse = mse(X_train.to_numpy(), Y_train.to_numpy(), coeff, len(Y_train)), \
                                        mse(X_test.to_numpy(), Y_test.to_numpy(), coeff, len(Y_test))
            print('{:^20f} | {:^20.3f} | {:^20.3f}'.format(lr, train_mse, validation_mse))

            if validation_mse < best_mse:
                best_lr, best_mse = lr, validation_mse
    print('\n\n')

    # (c) Use the validation data to pick the best converged solution, and report the learned weights for each feature.
    # Which features are the most important in deciding the house prices according to the learned weights? Compare them
    # to your pre-analysis results (Part 0 (d)).
    print('Part 1c: Report learned weights from the best converged solution\n')
    for column, coeff in zip(X_test.columns, coeffs[best_lr]):
        print('{:>13}: {:.3f}'.format(column, coeff))
