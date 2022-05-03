import pandas as pd
import matplotlib.pyplot as plt


def normalize_input(x, cols=None):
    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min())

    if cols is not None:
        for col in cols:
            x[col] = normalize(x[col])
    else:
        x = normalize(x)
        x.dummy.fillna(1, inplace=True)
    return x


def report_stats(X, num_cols, cat_cols):
    X_numerical_stats = X[num_cols].agg(['mean', 'std', 'min', 'max'])
    print('Numerical Feature Statistics\n{}\n'.format(X_numerical_stats))

    X_categorical_stats = pd.DataFrame(data={col: X[col].value_counts(normalize=True) * 100 for col in cat_cols})
    X_categorical_stats.fillna(0, inplace=True)
    print('Categorical Feature Statistics\n{}\n\n'.format(X_categorical_stats))


def preprocess_data(f_name, stats=False, normalize=True, num_cols=None, cat_cols=None, one_hot_cols=None,
                    norm_cols=None):
    # Load in training data
    df = pd.read_csv(f_name)

    # separate data into features and labels
    X = df[['dummy', 'id', 'date', 'bedrooms', 'bathrooms', 'sqft_living',
            'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
            'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
            'lat', 'long', 'sqft_living15', 'sqft_lot15']]
    Y = df['price']

    # Part 0 (20 pts) : Understanding your data + preprocessing. Perform the following steps:

    # (a) Remove the ID feature. Why do you think it is a bad idea to use this feature in learning?
    # Answer: it doesn't represent the data well, it has no correlation to the price of the house
    X.drop(columns=['id'], inplace=True)

    # (b) Split the date feature into three separate numerical features:
    # month, day , and year. Can you think of better ways of using this date feature?
    # Answer: Could be used to calculate the age of the house since the year built is provided
    X.insert(1, 'month', pd.DatetimeIndex(df['date']).month, True)
    X.insert(2, 'day', pd.DatetimeIndex(df['date']).day, True)
    X.insert(3, 'year', pd.DatetimeIndex(df['date']).year, True)
    X.drop(columns=['date'], inplace=True)

    # (c) Build a table that reports the statistics for each feature.
    # For numerical features, please report the mean, the standard deviation, and the range.
    # For categorical features such as waterfront, grade, condition (the later two are ordinal),
    # please report the percentage of examples for each category.
    if stats and num_cols is not None and cat_cols is not None:
        report_stats(X, num_cols, cat_cols)

    # one-hot encode categorical values
    X = pd.get_dummies(X, columns=one_hot_cols)

    # (e) Normalize all numerical features (excluding the housing prices y) to the range 0 and 1 using the training
    # data. This is equivalent to doing zi = xi−min(x) , where zi is the normalized feature for example xi,
    # max(x)−min(x) and max(x) and min(x) are the max/min over all examples for a specific feature. The normalization is
    # done on a per-feature basis. Note that when you apply the learned model from the normalized data to
    # test/validation data, you should make sure that you are using the same normalization procedure as used
    # in training. That is, when doing the scaling by max/min/mean anytime on your training data, you must save the
    # training max/min/mean for normalization on any other data. (If curious about normalization,
    # see https://www.sciencedirect.com/topics/computer-science/max-normalization)
    if normalize:
        X = normalize_input(X, norm_cols)

    return X, Y


if __name__ == '__main__':
    print('Part 0: Understanding your data + preprocessing\n')

    # for statistic reasons
    num_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view', 'sqft_above', 'sqft_basement',
                'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    cat_cols = ['month', 'day', 'year', 'waterfront', 'condition', 'grade', 'zipcode']

    # for preprocessing
    norm_cols = ['month', 'day', 'year', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
                 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
                 'sqft_living15', 'sqft_lot15']
    one_hot_cols = ['zipcode']

    X, Y = preprocess_data("PA1_train.csv", stats=True, normalize=True,
                           num_cols=num_cols, cat_cols=cat_cols, one_hot_cols=one_hot_cols, norm_cols=norm_cols)

    print(X.head(10))
