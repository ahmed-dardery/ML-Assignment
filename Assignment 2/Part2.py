"""
SVM
* First we need to convert the targets to be 1 and -1 only
* We need to find hyperplane such that wx + b = 0
* wx + b >= 1 if y = 1
* wx + b <= 1 if y = -1
* -> y*(wx + b)>=1

------

Cost Function
0 if y * f(x) >= 1
1 - y * f(x) else
so it can be cost = max(0,1-y*f(x))
where f(x) = wx + b

------

We have to maximize the margin which is 2/||w||

------


"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_data(file_path_csv, features, target_column='target'):
    data = pd.read_csv(file_path_csv)
    x = data[features]
    y = data[target_column]
    return x, y, data


# WX + b
def function(x, w):
    return np.dot(x, w)


def predict(x, w):
    return -1 if function(x, w) < 0 else 1


# Takes y {0,1} and returns y {-1,1}
def convert_y(y):
    return y * 2 - 1


def normalize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return np.divide(np.subtract(x, mu), sigma)


'''
1- Initialize weights
2- For every epoch loop on your data
3- For every datum calculate y * f(x) where f(x) = wx + b
4- If y * f(x) >= 1 -> correctly classified -> update w = w - alpha * 2 * lambda * w 
    (you don't need to update b because its derivative is zero)
5- Else -> misclassified -> update w = w + alpha * (y * x - 2 * lambda * w) and update b
'''


def svm(x, y, alpha, lmbda, epochs):
    w = np.zeros(x.shape[1])
    y = y.tolist()  # converts dictionary to array
    cost_history = []
    for _ in range(epochs):
        cost = 0
        for i, curr in enumerate(x):
            expr = y[i] * function(curr, w)
            if expr >= 1:
                w -= alpha * 2 * lmbda * w
            else:
                w += alpha * (np.dot(curr, y[i]) - 2 * lmbda * w)
                cost += 1 - expr

        cost_history.append(cost)
    return w, cost_history

def test(x, y, w):
    correct = 0
    y = y.tolist()  # converts dictionary to array
    for i, curr in enumerate(x):
        if y[i] == predict(curr, w):
            correct += 1
    return correct


def visualize_features(x, features, y):
    plt.scatter(x[features[0]], x[features[1]], marker='o', c=y)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.show()


def train_test_split(x, y, ratio):
    msk = np.random.rand(len(x)) < ratio
    return x[msk], y[msk], x[~msk], y[~msk]


# visualize different combination to see the best
def visualize_some_features(x, y):
    # visualize_features(x, ['age', 'trestbps'], y)
    # visualize_features(x, ['age', 'sex'], y)
    # visualize_features(x, ['age', 'cp'], y)
    # visualize_features(x, ['age', 'chol'], y)
    # visualize_features(x, ['age', 'chol'], y)
    # visualize_features(x, ['thalach', 'age'], y)
    # visualize_features(x, ['sex', 'oldpeak'], y)
    # visualize_features(x, ['exang', 'oldpeak'], y)
    # visualize_features(x, ['thalach', 'oldpeak'], y)
    visualize_features(x, ['exang', 'ca'], y)
    visualize_features(x, ['sex', 'ca'], y)
    visualize_features(x, ['sex', 'exang'], y)
    visualize_features(x, ['fbs', 'exang'], y)


def plot_history(history, c, label):
    plt.ylabel(label)
    plt.xlabel('Iteration')
    plt.plot(history, c=c, label=label)
    plt.legend()
    plt.show()


def main():
    heart_data = 'part2_data/heart.csv'
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                'thal']
    target_column = 'target'
    x, y, data = get_data(heart_data, features, target_column)
    # visualize_some_features(x, y)

    # train_features = ['age', 'chol']
    # train_features = ['age', 'thalach', 'chol']
    train_features = ['sex', 'exang', 'ca']

    y = convert_y(y)
    x = normalize(data[train_features])

    x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    bst = -1
    learning_rate = 0.004
    lmbda = 0.001
    epochs = 50
    avg = 0
    loop = 100
    best_history = []
    cnt = 0
    for i in range(loop):
        x_train, y_train, x_test, y_test = train_test_split(x, y, 0.8)
        w, cost_history = svm(x_train, y_train, learning_rate, lmbda, epochs)
        curr = test(x_test, y_test, w) / x_test.shape[0]
        avg += curr
        if curr > bst:
            cnt+=1
            best_history = cost_history
            bst = curr

    plot_history(best_history, 'orange', 'Cost History')
    print("Best result:", bst)
    print("Average:", avg / loop)


if __name__ == "__main__":
    main()
