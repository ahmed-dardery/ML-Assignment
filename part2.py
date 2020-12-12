import pandas as pd
import numpy as np
import shared as sh


# imported sklearn to compare my output with it ONLY
from sklearn.linear_model import LogisticRegression


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(x, theta):
    return sigmoid(np.dot(x, theta))


def compute_cost(theta, x, y):
    m = len(y)
    pred = predict(x, theta)
    j = y * np.log(pred) + (1 - y) * np.log(1 - pred)

    return np.sum(j) / (-m)


def train_logistic_regression(x, y, learning_rate, epochs):
    theta = np.zeros(x.shape[1])
    cost_history = [compute_cost(theta, x, y)]
    accuracy_history = [calculate_accuracy(x, y, theta)]
    for i in range(epochs):
        h = predict(x, theta)
        gradient = sh.calculate_gradient(x, h, y)
        theta = sh.update_theta(theta, learning_rate, gradient)
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


def main():
    heart_data = 'data/heart.csv'
    features = ['trestbps', 'chol', 'thalach', 'oldpeak']
    target_column = 'target'
    learning_rate = 0.1
    epochs = 200

    x, y = sh.get_data(heart_data, features, target_column)

    mu, sigma = sh.get_normalization(x)
    x_norm = sh.apply_normalization(x, mu, sigma)

    x_norm = sh.add_intercepts(x_norm)

    train_x, train_y = x_norm, y
    test_x, test_y = x_norm, y

    theta, cost_history, accuracy_history = train_logistic_regression(train_x, train_y, learning_rate, epochs)

    sh.plot_history(cost_history, "orange", "Cost Function")
    sh.plot_history(accuracy_history, "green", "Accuracy")

    hypothesis = "S(" + sh.get_hypothesis(theta) + ")"

    print("Normalization:")
    print("mu:", mu, sep='\n')
    print()
    print("sigma:", sigma, sep='\n')
    print()
    print("Hypothesis: ", hypothesis)

    print()
    print("Our model's accuracy: ", calculate_accuracy(test_x, test_y, theta, target_column))
    print("Sklearn's accuracy: ", sklearn_accuracy(sklearn_model(x, y), x, y, target_column))


if __name__ == "__main__":
    main()
