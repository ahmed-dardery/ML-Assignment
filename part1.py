import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def predict(x, theta):
    return np.matmul(x, theta)


def normalize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return mu, sigma


def apply_normalization(x, mu, sigma):
    return np.divide(np.subtract(x, mu), sigma)


def add_ones(x):
    sz = x.shape[0]
    return np.append(np.ones((sz, 1)).astype(int), x, axis=1)


def compute_cost(theta, x, y):
    m = len(y)
    y_pred = predict(x, theta)
    cost = np.square(np.subtract(y, y_pred))

    return np.sum(cost) / (2 * m)


def next_theta(alpha, theta, x, y):
    m = len(y)
    y_pred = np.matmul(x, theta)

    err = np.subtract(y_pred, y)
    d_theta = np.matmul(x.T, err)
    return theta - (alpha / m) * d_theta


def gradient_descend(iter, alpha, x, y):
    theta = np.zeros((x.shape[1], 1))
    cost_history = [compute_cost(theta, x, y)]
    for _ in range(iter):
        theta = next_theta(alpha, theta, x, y)
        cost_history.append(compute_cost(theta, x, y))
    return cost_history, theta


def get_hypothesis(theta):
    th = theta.T[0]
    hyp = format(th[0], ".3f")
    if len(th) > 1:
        hyp += "".join(" + {v:.3f} * X{i}".format(v=v, i=i + 1) for i, v in enumerate(th[1:]))
    return hyp


def plot_data(x, y, y_pred, x_label, y_label, extra_label):
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.plot(x, y_pred, c='r', label=extra_label)
    plt.scatter(x, y, s=0.3, label='Input Data')
    plt.legend()
    plt.show()


def plot_cost(cost_history):
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration')
    plt.plot(cost_history, c='orange', label='Cost Function')
    plt.legend()
    plt.show()


def main():
    house_data = pd.read_csv('data/house_data.csv')
    features = ['sqft_living']
    features = ['grade', 'bathrooms', 'lat', 'sqft_living', 'view']
    x_org = house_data[features]
    y = house_data[['price']]

    y = y.to_numpy() / (10 ** 3)

    mu, sigma = normalize(x_org)
    x = add_ones(apply_normalization(x_org, mu, sigma))
    x_org = add_ones(x_org)

    cost_history, theta = gradient_descend(200, 0.1, x, y)
    hypothesis = get_hypothesis(theta)

    print("Normalization:")
    print("mu:", mu, sep='\n')
    print()
    print("sigma:", sigma, sep='\n')
    print()
    print("Hypothesis:", hypothesis)

    if len(features) == 1:
        y_pred = predict(x, theta)
        plot_data(x_org[:, 1], y, y_pred, 'Price', 'Living room area in thousands of sqft (1000)', hypothesis)

    plot_cost(cost_history)


if __name__ == "__main__":
    main()
