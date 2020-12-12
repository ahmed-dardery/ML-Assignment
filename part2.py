import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# imported sklearn to compare my output with it ONLY
from sklearn.linear_model import LogisticRegression

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
    cost_history = [compute_cost(theta, x, y)]
    accuracy_history = [calculate_accuracy(x, y, theta)]
    for i in range(epochs):
        h = predict(x, theta)
        gradient = calculate_gradient(x, h, y)
        theta = update(theta, learning_rate, gradient)
        accuracy_history.append(calculate_accuracy(x, y, theta))
        cost_history.append(compute_cost(theta, x, y))

    return theta, cost_history, accuracy_history


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


def plot_history(history, c, label):
    plt.ylabel(label)
    plt.xlabel('Iteration')
    plt.plot(history, c=c, label=label)
    plt.legend()
    plt.show()


def compute_cost(theta, x, y):
    m = len(y)
    pred = predict(x, theta)
    j = y * np.log(pred) + (1 - y) * np.log(1 - pred)

    return np.sum(j) / (-m)


def get_hypothesis(theta):
    th = theta
    hyp = format(th[0], ".3f")
    if len(th) > 1:
        hyp += "".join(" + {v:.3f} * X{i}".format(v=v, i=i + 1) for i, v in enumerate(th[1:]))
    return hyp


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

    theta, cost_history, accuracy_history = train_logistic_regression(train_x, train_y, learning_rate, epochs)

    plot_history(cost_history, "orange", "Cost Function")
    plot_history(accuracy_history, "green", "Accuracy")

    skmodel = sklearn_model(x, y)

    hypothesis = "S(" + get_hypothesis(theta) + ")"

    print("Normalization:")
    print("mu:", mu, sep='\n')
    print()
    print("sigma:", sigma, sep='\n')
    print()
    print("Hypothesis: ", hypothesis)

    print()
    print("Our model's accuracy: ", calculate_accuracy(test_x, test_y, theta, target_column))
    print("Sklearn's accuracy: ", sklearn_accuracy(skmodel, x, y, target_column))


if __name__ == "__main__":
    main()
