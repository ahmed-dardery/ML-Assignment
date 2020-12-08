import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# imported sklearn to compare my output with it ONLY
from sklearn.linear_model import LogisticRegression


def sigmoid(x, weight):
    z = np.dot(x, weight)
    return 1 / (1 + np.exp(-z))


def log_likelihood(x, y, weights):
    theta_x = np.dot(x, weights)
    ll = np.sum(y * theta_x - np.log(1 + np.exp(theta_x)))
    return ll


def calculate_gradient(x, h, y):
    return np.dot(x.T, (h - y)) / y.shape[0]


def update(theta, alpha, gradient):
    return theta - alpha * gradient


def normalize(df):
    df_cpy = df.copy()
    for column in df_cpy.columns:
        df_cpy[column] = (df_cpy[column] - df_cpy[column].mean()) / df_cpy[column].std()

    return df_cpy


def predict(x, theta):
    return sigmoid(x, theta)


def add_intercepts(x):
    return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

# if you remove add_intercepts from 45 and 58 and replace them with ``train_x = x , test_data = x`` accuracy gets better
def train_logistic_regression(x, y, learning_rate, epochs):
    train_x = add_intercepts(x)
    theta = np.zeros(train_x.shape[1])

    for i in range(epochs):
        print(calculate_accuracy(x, y, theta))
        h = sigmoid(train_x, theta)
        gradient = calculate_gradient(train_x, h, y)
        theta = update(theta, learning_rate, gradient)

    return theta


def calculate_accuracy(x, targets, theta, target_column='target'):
    test_data = add_intercepts(x)
    prediction = predict(test_data, theta)
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


def main():
    heart_data = 'heart.csv'
    features = ['trestbps', 'chol', 'thalach', 'oldpeak']
    target_column = 'target'
    learning_rate = 0.01
    epochs = 10

    x, y = get_data(heart_data, features, target_column)

    normalized_x = normalize(x)
    train_x, train_y = normalized_x, y
    test_x, test_y = normalized_x, y

    theta = train_logistic_regression(train_x, train_y, learning_rate, epochs)

    print(calculate_accuracy(test_x, test_y, theta, target_column))

    skmodel = sklearn_model(x, y)
    print(sklearn_accuracy(skmodel, x, y, target_column))


if __name__ == "__main__":
    main()
