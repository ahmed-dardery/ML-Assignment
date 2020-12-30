import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_data(file_path_csv, features, target_column='target'):
    data = pd.read_csv(file_path_csv)
    x = data[features]
    y = data[target_column]
    return x, y


def calculate_gradient(x, h, y):
    return np.dot(x.T, (h - y)) / y.shape[0]


def update_theta(theta, alpha, gradient):
    return theta - alpha * gradient


def get_normalization(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return mu, sigma


def apply_normalization(x, mu, sigma):
    return np.divide(np.subtract(x, mu), sigma)


def add_intercepts(x):
    return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)


def plot_history(history, c, label):
    plt.ylabel(label)
    plt.xlabel('Iteration')
    plt.plot(history, c=c, label=label)
    plt.legend()
    plt.show()


def get_hypothesis(theta):
    th = theta
    hyp = format(th[0], ".3f")
    if len(th) > 1:
        hyp += "".join(" + {v:.3f} * X{i}".format(v=v, i=i + 1) for i, v in enumerate(th[1:]))
    return hyp


def plot_data(x, y, y_pred, x_label, y_label, line_label, scatter_label="Input Data"):
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.plot(x, y_pred, c='r', label=line_label)
    plt.scatter(x, y, s=0.3, label=scatter_label)
    plt.legend()
    plt.show()