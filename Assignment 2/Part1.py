import numpy as np
import pandas as pd


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


def calculate_information_gain(x, y):
    children, details = np.unique(x, return_inverse=True)

    avg = 0
    for j in range(len(children)):
        child = y[details == j]
        entropy_child = calculate_entropy(child)

        avg += entropy_child * len(child) / len(y)

    return calculate_entropy(y) - avg

def recurse(df, parent):
    pass

def generate_decision_tree(df):
    parent = None
    return recurse(df, parent)


def solve(df):
    print(df)
    n, m = df.shape
    for i in range(1, m):
        print(calculate_information_gain(df.iloc[:, i], df.iloc[:, 0]))
        # cur_col_y = cur_col[cur_col[i] == 'y']
        # yes = cur_col_y[0].isin([POS]).sum(axis=0)
        # no = n - yes

    # print(df.loc[df.iloc[:, 0] == NEG])


# Fills missing data points by majority of the row
def fill_in_unknowns(df):
    x = df.iloc[:, 1:]
    replacement = (x.isin(['y']).sum(axis=0) >= x.isin(['n']).sum(axis=0)).tolist()
    for i in range(x.shape[1]):
        x.iloc[:, i].replace('?', 'y' if replacement[i] else 'n', inplace=True)


def main():
    df = pd.read_csv('part1_data/house-votes-84.data.txt', header=None)
    train, test = split_to_train_test(df, 0.01)
    fill_in_unknowns(train)
    solve(train)


if __name__ == '__main__':
    main()
