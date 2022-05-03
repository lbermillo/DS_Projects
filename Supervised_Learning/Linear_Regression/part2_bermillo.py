import numpy as np

from part0_bermillo import preprocess_data
from part1_bermillo import train


# Part 2 (20 pts). Training with non-normalized data Use the preprocessed data but skip the normalization. Consider
# at least the following values for learning rate: 100, 10, 1E-1 , 1E-2 , 1E-3 , 1E-4 . For each value, train up to
# 10000 iterations ( Fix the number of iterations for this part). If training is clearly diverging, you can terminate
# early. Plot the training MSE and validation MSE respectively as a function of the number of iterations. What do you
# observe? Specify the learning rate value (if any) that prevents the gradient descent from exploding? Compare between
# using the normalized and the non-normalized versions of the data. Which one is easier to train and why?

if __name__ == '__main__':
    # for preprocessing
    one_hot_cols = ['zipcode']

    X_train, Y_train = preprocess_data("PA1_train.csv", normalize=False, one_hot_cols=one_hot_cols)

    # dictionaries to track coefficients and losses for each learning rate
    coeffs, losses = {}, {}

    print('Part 2: Training with non-normalized data\n')

    # run training for each learning rate
    for lr in [100, 10, 1e-1, 1e-2, 1e-3, 1e-4, 1e-15]:
        train_coeff, train_losses = train(X_train.to_numpy(), Y_train.to_numpy(), lr=lr, epochs=1e4)

        # add training losses and coefficients to dictionary with their corresponding learning rate
        coeffs[lr], losses[lr] = train_coeff, train_losses

        print('{:^20} | {:^20.3f} | {:^20}'.format(lr, train_losses[-1], len(train_losses)))