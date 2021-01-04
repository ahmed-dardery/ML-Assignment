import numpy as np
import pandas as pd


class TreeNode:
    def __init__(self, lbl):
        self.label = lbl
        self.nxt = {}

    def predict(self, x):
        if x[self.label] not in self.nxt:
            return 'unpredictable'

        go_next = self.nxt[x[self.label]]
        if not isinstance(go_next, TreeNode):
            return go_next
        return go_next.predict(x)


def split_to_train_test(df, train_ratio):
    train = df.sample(frac=train_ratio)
    test = df.drop(train.index)
    return train, test


def calculate_entropy(y):
    classes, count = np.unique(y, return_counts=True)
    size = len(y)
    entropy_value = np.sum(
        [(-count[i] / size) * np.log2(count[i] / size) for i in range(len(classes))]
    )
    return entropy_value


def calculate_information_gain(feat, y):
    children, details = np.unique(feat, return_inverse=True)

    avg = 0
    for j in range(len(children)):
        child = y[details == j]
        entropy_child = calculate_entropy(child)

        avg += entropy_child * len(child) / len(y)

    return calculate_entropy(y) - avg


def recurse(x, y, features):
    unique, cnt = np.unique(y, return_counts=True)
    if len(unique) <= 1:
        return unique[0]
    if len(features) == 0:
        return unique[np.argmax(cnt)]

    gains = [calculate_information_gain(x[feat], y) for feat in features]
    optimal = features[np.argmax(gains)]
    node = TreeNode(optimal)
    for choice in np.unique(x[optimal]):
        subset = x[optimal] == choice
        node.nxt[choice] = recurse(x[subset], y[subset], [v for v in features if v != optimal])

    return node


def generate_decision_tree(x, y):
    return recurse(x, y, features=x.columns.tolist())


# Fills missing data points by majority of the row
def fill_in_unknowns(df, majority):
    x = df.iloc[:, 1:]
    for i in range(x.shape[1]):
        x.iloc[:, i].replace('?', 'y' if majority[i] else 'n', inplace=True)


def find_majority(df):
    x = df.iloc[:, 1:]
    return (x.isin(['y']).sum(axis=0) >= x.isin(['n']).sum(axis=0)).tolist()


def calculate_accuracy(tree, df):
    total = 0
    for i in range(len(df)):
        if tree.predict(df.iloc[i, 1:]) == df.iloc[i, 0]:
            total += 1
    return total / len(df)


def testing():
    train = pd.read_csv('part1_data/lecture.txt', header=None)
    tree = generate_decision_tree(train.iloc[:, 1:], train.iloc[:, 0])
    print("Training accuracy: ", calculate_accuracy(tree, train))


def main():
    df = pd.read_csv('part1_data/house-votes-84.data.txt', header=None)
    train, test = split_to_train_test(df, 0.25)
    # majority = find_majority(df)
    # fill_in_unknowns(df, majority)
    majority = find_majority(train)
    fill_in_unknowns(train, majority)
    fill_in_unknowns(test, majority)
    tree = generate_decision_tree(train.iloc[:, 1:], train.iloc[:, 0])

    print("Training accuracy: ", calculate_accuracy(tree, train))
    print("Testing  accuracy: ", calculate_accuracy(tree, test))


if __name__ == '__main__':
    main()
