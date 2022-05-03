import numpy as np

from part0_bermillo import preprocess_data, normalize_input
from part1_bermillo import train, mse

# Part 3 (10 pts). Feature engineering (exploration) Similar to Part 0(b), list any modifications to the features
# provided that you think may be useful for making better predictions. Implement 3 (or more) ideas you have, and report
# the results. This is an open question, so you are free to do your own exploration. For example, one simple extension
# is to try the two ways of representing the zip code feature (numerical or categorical), and use the opposite of what
# you have been for the assignment thus far. Also, you may try combinations of features (such as ratios, products,
# polynomial features, etc). Please report what you have tried and what (if any) effect it has on the predictions.

# 3. remove some redundancies (sqft_living15, sqft_lot15, lat, long)
# 4. normalize all features (numerical and categorical)

if __name__ == '__main__':
    norm_cols = ['month', 'day', 'year', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
                 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
                 'sqft_living15', 'sqft_lot15']
    one_hot_cols = ['zipcode']

    X_train, Y_train = preprocess_data("PA1_train.csv", normalize=False, one_hot_cols=one_hot_cols)
    X_test, Y_test   = preprocess_data("PA1_dev.csv", normalize=False, one_hot_cols=one_hot_cols)

    # 1. age of the house:  year - year_built
    # 2. take the product of grade and condition to consolidate: score = grade * condition
    new_features = {'age': X_train.year - X_train.yr_built,
                    'score': X_train.grade * X_train.condition, }

    lr = 1e-4

    print('{:^20} | {:^20} | {:^20} | {:^20}'.format('Feature', 'Training Loss', 'Epochs', 'Validation Loss'))

    # control
    norm_X_train, norm_X_test = normalize_input(X_train.copy(), norm_cols), normalize_input(X_test.copy(), norm_cols)
    coeff_control, losses_control = train(norm_X_train.to_numpy(), Y_train.to_numpy(), lr=lr, epsilon=1e-3)
    validation_control = mse(norm_X_test.to_numpy(), Y_test.to_numpy(), coeff_control, len(Y_test))
    print('{:^20} | {:^20.3f} | {:^20} | {:^20.3f}'.format('no new features', losses_control[-1], len(losses_control),
                                                           validation_control))

    # norm zipcode
    norm_X_train, norm_X_test = normalize_input(X_train.copy()), normalize_input(X_test.copy())
    coeff_all_norm, losses_all_norm = train(norm_X_train.to_numpy(), Y_train.to_numpy(), lr=lr, epsilon=1e-3)
    validation_control = mse(norm_X_test.to_numpy(), Y_test.to_numpy(), coeff_all_norm, len(Y_test))
    print('{:^20} | {:^20.3f} | {:^20} | {:^20.3f}'.format('all norm features',
                                                           losses_all_norm[-1],
                                                           len(losses_all_norm),
                                                           validation_control))

    for feature, values in new_features.items():
        columns_to_normalize = ['month', 'day', 'year', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                                'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
                                'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15', feature]

        new_X_train, new_X_test = X_train.copy(), X_test.copy()
        new_X_train[feature], new_X_test[feature] = values, values

        new_X_train, new_X_test = normalize_input(new_X_train, columns_to_normalize), \
                                  normalize_input(new_X_test, columns_to_normalize)

        coeff, losses = train(new_X_train.to_numpy(), Y_train.to_numpy(), lr=lr, epsilon=1e-3)
        validation = mse(new_X_test.to_numpy(), Y_test.to_numpy(), coeff, len(Y_test))

        print('{:^20} | {:^20.3f} | {:^20} | {:^20.3f}'.format(feature, losses[-1], len(losses), validation))


    # 3. remove some redundancies ( sqft_living15, sqft_lot15, lat and long (can be represented by zipcode) )
    new_X_train, new_X_test = X_train.copy(), X_test.copy()
    new_X_train.drop(columns=['sqft_living15', 'sqft_lot15', 'lat', 'long', 'condition'], inplace=True)
    new_X_test.drop(columns=['sqft_living15', 'sqft_lot15', 'lat', 'long', 'condition'], inplace=True)

    norm_cols = ['month', 'day', 'year', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
                 'view', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
    new_X_train, new_X_test = normalize_input(new_X_train, norm_cols), normalize_input(new_X_test, norm_cols)

    coeff, losses = train(new_X_train.to_numpy(), Y_train.to_numpy(), lr=lr, epsilon=1e-3)
    validation = mse(new_X_test.to_numpy(), Y_test.to_numpy(), coeff, len(Y_test))

    print('{:^20} | {:^20.3f} | {:^20} | {:^20.3f}'.format('remove redundancies', losses[-1], len(losses), validation))

