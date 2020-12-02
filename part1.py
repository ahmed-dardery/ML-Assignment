import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_cost(theta, X, Y):
    m = len(Y)
    pred = np.matmul(X, theta)
    J = np.square(np.subtract(Y, pred))

    return np.sum(J) / (2 * m)


def next_theta(alpha, theta, X, Y):
    m = len(Y)
    pred = np.matmul(X, theta)

    err = np.subtract(pred, Y)
    dtheta = np.matmul(X.T, err)
    return theta - (alpha / m) * dtheta


def gradient_descend(iter, alpha, X, Y):
    theta = np.zeros((X.shape[1], 1))
    JHistory = [compute_cost(theta, X, Y)]
    for _ in range(iter):
        theta = next_theta(alpha, theta, X, Y)
        JHistory.append(compute_cost(theta, X, Y))
    return JHistory, theta


def get_hypothesis(theta):
    th = theta.T[0]
    hyp = format(th[0], ".3f")
    if len(th) > 1:
        hyp += "".join(" + {v:.3f} * X{i}".format(v=v, i=i + 1) for i, v in enumerate(th[1:]))
    return hyp


def plot_data(X, Y, theta, xlabel, ylabel):
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.plot(X[:, 1], np.matmul(X, theta), c='r', label=hypothesis)
    plt.scatter(X[:, 1], Y, s=0.3, label='Input Data')
    plt.legend()
    plt.show()


def plot_cost(JHistory):
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration')
    plt.plot(JHistory, c='orange', label='Cost Function')
    plt.legend()
    plt.show()


houseData = pd.read_csv('data/house_data.csv')
features = ['sqft_living']
# features = ['grade', 'bathrooms', 'lat', 'sqft_living', 'view']
X = houseData[features]
Y = houseData[['price']]

allM = len(Y)
Y = Y.to_numpy() / (10 ** 3)
X = X / (10 ** 3)
X = np.append(np.ones((allM, 1)).astype(int), X, axis=1)

JHistory, theta = gradient_descend(100, 0.01, X, Y)
hypothesis = get_hypothesis(theta)

if len(features) == 1:
    plot_data(X, Y, theta, 'Price in thousands of dollars (1000)', 'Living room area in thousands of sqft (1000)')

plot_cost(JHistory)
