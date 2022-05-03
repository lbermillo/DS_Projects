import numpy as np
import pandas as pd


def create_node(name=None, gain=None, branch_0=None, branch_1=None):
    return {'feature': name, 'info_gain': gain, 0: branch_0, 1: branch_1}


def calc_entropy(n_pos, n_neg):
    # if one of the classes have zero examples, then the enropy is 0
    if n_pos == 0 or n_neg == 0:
        return 0

    # get total number of examples
    N = n_pos + n_neg

    return -np.sum([(n_pos / N) * np.log2(n_pos / N), (n_neg / N) * np.log2(n_neg / N)])


def calc_information_gain(data, feature, root_entropy, N):
    # split into the feature's branches
    pos_branch, neg_branch = data[data[feature] == 1], data[data[feature] == 0]

    # get pos and neg labels from each branch
    pos_pos_branch, neg_pos_branch = pos_branch[pos_branch['Response'] == 1], pos_branch[pos_branch['Response'] == 0]
    pos_neg_branch, neg_neg_branch = neg_branch[neg_branch['Response'] == 1], neg_branch[neg_branch['Response'] == 0]

    # calculate the entropy for each branch
    pos_branch_entropy = calc_entropy(len(pos_pos_branch), len(neg_pos_branch))
    neg_branch_entropy = calc_entropy(len(pos_neg_branch), len(neg_neg_branch))

    # calculate conditional entropy for this feature
    feature_entropy = np.sum([(1 / N) * len(pos_branch) * pos_branch_entropy,
                              (1 / N) * len(neg_branch) * neg_branch_entropy])

    return np.round(root_entropy - feature_entropy, decimals=3)


def predict(data, decision_tree):
    # get the feature of the root
    feature = decision_tree['feature']

    # match to example
    result = decision_tree[data[feature]]

    while result not in [0, 1]:
        # get the feature from the node
        feature = result['feature']

        # match to example
        result = result[data[feature]]

    return result


def build_random_tree(data, root, features, m, depth, dmax, prev_pred):
    # base case 1: branch has no examples
    if len(data) == 0:
        return prev_pred

    # base case 2: leaf node (all labels are the same or pure)
    elif len(set(data['Response'])) == 1:
        return data['Response'].values[0]

    # base case 3: reached max depth
    elif depth >= dmax:
        # return label with the highest count
        return data['Response'].value_counts().idxmax()

    # check if a root already exists
    if root is None:
        # initalize a root node
        root = create_node()

    # split data by their labels
    pos, neg = data[data['Response'] == 1], data[data['Response'] == 0]

    # 1. calculate the root node's entropy
    root_entropy = calc_entropy(len(pos), len(neg))

    # 2. Sample m features from the feature list
    m_features = np.random.choice(features, m, replace=False)

    # 3. calculate information gain for each feature and
    information_gains = {feature: calc_information_gain(data, feature, root_entropy, len(data)) for feature in
                         m_features}

    # 4. choose feature with highest information gain to be the next root
    root['feature'] = max(information_gains, key=information_gains.get)
    root['info_gain'] = information_gains[root['feature']]

    # 5. go to the lhs of the tree
    root[0] = build_random_tree(data[data[root['feature']] == 0], root[0], features, m, depth + 1, dmax,
                                data['Response'].value_counts().idxmax())

    # 6. go to the rhs of the tree
    root[1] = build_random_tree(data[data[root['feature']] == 1], root[1], features, m, depth + 1, dmax,
                                data['Response'].value_counts().idxmax())

    return root


def build_random_forest(data, m, dmax, T):
    # extract feature names from dataframe
    features = data.drop(columns=['Response']).columns

    # return built tree
    return [
        build_random_tree(data=data.sample(frac=1, replace=True), root=None, features=features, m=m, depth=0, dmax=dmax,
                          prev_pred=data['Response'].value_counts().idxmax())
        for _ in range(T)]


def majority_vote(data, random_forest):
    # return majority vote for each example
    return pd.concat([data.apply(predict, decision_tree=tree, axis=1)
                      for tree in random_forest], axis=1).mode(axis=1)[0]


def accuracy(y_hat, y):
    return (np.sum(y_hat == y) / len(y)) * 100


if __name__ == '__main__':

    # Load data in
    train_data = pd.read_csv('pa4_data/pa4_train_X.csv')
    test_data = pd.read_csv('pa4_data/pa4_dev_X.csv')

    # Add labels to data
    train_data['Response'] = pd.read_csv('pa4_data/pa4_train_y.csv', header=None).values.astype(int).flatten()
    test_data['Response'] = pd.read_csv('pa4_data/pa4_dev_y.csv', header=None).values.astype(int).flatten()

    for dmax in [2, 10, 25]:

        # run experiments
        train_accuracies, test_accuracies = {}, {}

        # the number of features to sub-sample in each test selection step.
        for m in [5, 25, 50, 100]:
            # set seed to keep consistent results
            np.random.seed(1)

            # the number of trees to include in your random forest.
            random_forest = build_random_forest(train_data, m=m, dmax=dmax, T=100)

            # get accuracies
            train_accuracies[m] = {
                T: accuracy(majority_vote(train_data, random_forest[:int(T)]), train_data['Response'])
                for T in np.linspace(10, 100, 10)}
            test_accuracies[m] = {T: accuracy(majority_vote(test_data, random_forest[:int(T)]), test_data['Response'])
                                  for T
                                  in np.linspace(10, 100, 10)}

            print(
                'Depth: {:2} \n'
                    '\tSub-samples (m): {:2} \n'
                        '\t\tTrain Accuracies: \n'
                            '\t\t\t{} \n'
                        '\t\t Validation Accuracies: \n'
                            '\t\t\t{}\n'.format(int(dmax),
                                                m,
                                                train_accuracies[m],
                                                test_accuracies[m]), )
