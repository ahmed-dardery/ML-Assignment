import numpy as np
import pandas as pd


def split_to_train_test(df, train_ratio):
    train = df.sample(frac=train_ratio)
    test = df.drop(train.index)
    return train, test


# Fills missing data points by majority of the row
def fill_in_unknowns(x):
    replacement = x.isin(['y']).sum(axis=0) >= x.isin(['n']).sum(axis=0)
    for i in range(x.shape[1]):
        x.iloc[:, i].replace('?', 'y' if replacement[i] else 'n', inplace=True)


def main():
    df = pd.read_csv('part1_data/house-votes-84.data.txt')
    train, test = split_to_train_test(df, 0.25)
    x = train.iloc[:, 1:]
    y = train.iloc[:, 0]
    fill_in_unknowns(x)


if __name__ == '__main__':
    main()
