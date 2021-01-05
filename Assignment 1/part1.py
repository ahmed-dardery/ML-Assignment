import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shared as sh

# imported sklearn to compare my output with it ONLY
from sklearn.linear_model import LinearRegression

def predict(x, theta):
    return np.matmul(x, theta)


def compute_cost(theta, x, y):
    m = len(y)
    y_pred = predict(x, theta)
    cost = (y - y_pred) ** 2

    return np.sum(cost) / (2 * m)


def gradient_descend(iter, alpha, x, y):
    theta = np.zeros(x.shape[1])
    cost_history = [compute_cost(theta, x, y)]
    for _ in range(iter):
        h = predict(x, theta)
        gradient = sh.calculate_gradient(x, h, y)
        theta = sh.update_theta(theta, alpha, gradient)
        cost_history.append(compute_cost(theta, x, y))

    return theta, cost_history


def sklearn_model(x, y):
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    return model


def sklearn_error(model, test_x, test_y):
    y_pred = model.predict(test_x)
    return np.sum((test_y - y_pred) ** 2) / (2 * len(test_y))


def main():
    heart_data = 'data/house_data.csv'
    features = ['sqft_living']
    #features = ['grade', 'bathrooms', 'lat', 'sqft_living', 'view']
    target_column = 'price'
    learning_rate = 0.5
    epochs = 100

    x_org, y = sh.get_data(heart_data, features, target_column)

    y = y.to_numpy() / (10 ** 3)
    #y = y.to_numpy()
    mu, sigma = sh.get_normalization(x_org)
    x = sh.add_intercepts(sh.apply_normalization(x_org, mu, sigma))
    x_org = sh.add_intercepts(x_org)

    theta, cost_history = gradient_descend(epochs, learning_rate, x, y)
    hypothesis = sh.get_hypothesis(theta)

    print("Normalization:")
    print("mu:", mu, sep='\n')
    print()
    print("sigma:", sigma, sep='\n')
    print()
    print("Hypothesis:", hypothesis)

    if len(features) == 1:
        y_pred = predict(x, theta)
        sh.plot_data(x_org[:, 1], y, y_pred, 'Living room area in thousands of sqft (1000)', 'Price', hypothesis)
    else:
        y_pred = predict(x, theta)
        sh.plot_data(y_pred, y, y_pred, 'Predicted Price', 'Price', "Hypothesis", "Input Prediction")
    print()
    print("Our model's error: ", compute_cost(theta, x, y))
    print("Sklearn's error: ", sklearn_error(sklearn_model(x, y), x, y))

    sh.plot_history(cost_history, "orange", "Cost Function")


if __name__ == "__main__":
    main()
