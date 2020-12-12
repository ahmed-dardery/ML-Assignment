import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# imported sklearn to compare my output with it ONLY
from sklearn.linear_model import LogisticRegression


def log_likelihood(x, y, weights):
    theta_x = np.dot(x, weights)
    ll = np.sum(y * theta_x - np.log(1 + np.exp(theta_x)))
    return ll


def calculate_gradient(x, h, y):
    return np.dot(x.T, (h - y)) / y.shape[0]


def update(theta, alpha, gradient):
    return theta - alpha * gradient


def normalize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return mu, sigma


def apply_normalization(x, mu, sigma):
    return np.divide(np.subtract(x, mu), sigma)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(x, theta):
    return sigmoid(np.dot(x, theta))


def add_intercepts(x):
    return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

# if you remove add_intercepts from 45 and 58 and replace them with ``train_x = x , test_data = x`` accuracy gets better
def train_logistic_regression(x, y, learning_rate, epochs):
    theta = np.zeros(x.shape[1])
    cost_history = []

    for i in range(epochs):
        print(calculate_accuracy(x, y, theta))
        h = predict(x, theta)
        gradient = calculate_gradient(x, h, y)
        theta = update(theta, learning_rate, gradient)
        cost_history.append(compute_cost(theta, x, y))

    return theta, cost_history


def calculate_accuracy(x, targets, theta, target_column='target'):
    prediction = predict(x, theta)
    df_prediction = pd.DataFrame(np.around(prediction)).join(targets)
    df_prediction['predicts'] = df_prediction[0].apply(lambda x: 0 if x < 0.5 else 1)
    return (df_prediction.loc[df_prediction['predicts'] == df_prediction[target_column]].shape[0] /
            df_prediction.shape[0] * 100)


def sklearn_model(x, y):
    model = LogisticRegression(fit_intercept=True, max_iter=100000)
    model.fit(x, y)
    return model


def sklearn_accuracy(model, test_x, test_y, target_column='target'):
    prediction = model.predict(test_x)
    df_prediction = pd.DataFrame(prediction).join(test_y)
    return df_prediction.loc[df_prediction[0] == df_prediction[target_column]].shape[0] / df_prediction.shape[0] * 100


def get_data(file_path_csv, features, target_column='target'):
    data = pd.read_csv(file_path_csv)
    x = data[features]
    y = data[target_column]
    return x, y


def plot_cost(cost_history):
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration')
    plt.plot(cost_history, c='orange', label='Cost Function')
    plt.legend()
    plt.show()


def compute_cost(theta, x, y):
    m = len(y)
    pred = predict(x, theta)
    j = y * np.log(pred) + (1 - y) * np.log(1 - pred)

    return np.sum(j) / (-m)


def main():
    heart_data = 'data/heart.csv'
    features = ['trestbps', 'chol', 'thalach', 'oldpeak']
    target_column = 'target'
    learning_rate = 0.01
    epochs = 10

    x, y = get_data(heart_data, features, target_column)

    mu, sigma = normalize(x)
    x_norm = apply_normalization(x, mu, sigma)

    x_norm = add_intercepts(x_norm)

    train_x, train_y = x_norm, y
    test_x, test_y = x_norm, y

    theta, cost_history = train_logistic_regression(train_x, train_y, learning_rate, epochs)

    plot_cost(cost_history)

    print(calculate_accuracy(test_x, test_y, theta, target_column))

    skmodel = sklearn_model(x, y)
    print(sklearn_accuracy(skmodel, x, y, target_column))


if __name__ == "__main__":
    main()
