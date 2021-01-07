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
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def get_data(file_path_csv, features, target_column='target'):
    data = pd.read_csv(file_path_csv)
    x = data[features]
    y = data[target_column]
    return x, y, data


# WX + b
def function(x, w, b):
    return np.dot(x, w) + b


def predict(x, w, b):
    return -1 if function(x, w, b) < 0 else 1


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
    b = 0
    y = y.tolist() #converts dictionary to array
    for _ in range(epochs):
        for i, curr in enumerate(x.values):
            if y[i] * function(curr, w, b) >= 1:
                w -= alpha * 2 * lmbda * w
            else:
                w += alpha * (np.dot(curr, y[i]) - 2 * lmbda * w)
                b += alpha * y[i]
    return w, b


def test(x, y, w, b):
    correct = 0
    y = y.tolist() #converts dictionary to array
    for i, curr in enumerate(x.values):
        if y[i] == predict(curr, w, b):
            correct += 1
    return correct


def visualize_features(X, features, y):
    plt.scatter(X[features[0]], X[features[1]], marker='o', c=y)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.show()


def train_test_split(x, y, ratio):
    msk = np.random.rand(len(x)) < ratio
    return x[msk], y[msk], x[~msk], y[~msk]


def main():
    heart_data = 'part2_data/heart.csv'
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                'thal']
    target_column = 'target'

    train_features = ['sex', 'exang', 'ca']
    # train_features = ['sex', 'exang', 'ca', 'fbs']
    #train_features = ['thalach', 'trestbps']
    # visualize different combination to see the best
    # visualize_features(x, ['age', 'trestbps'], y)
    # visualize_features(x, ['age', 'sex'], y)
    # visualize_features(x, ['age', 'cp'], y)
    # visualize_features(x, ['age', 'chol'], y)
    # visualize_features(x, ['age', 'chol'], y)
    # visualize_features(x, ['thalach', 'age'], y)
    # visualize_features(x, ['sex', 'oldpeak'], y)
    # visualize_features(x, ['exang', 'oldpeak'], y)
    # visualize_features(x, ['thalach', 'oldpeak'], y)
    # visualize_features(x, ['exang', 'ca'], y)
    learning_rate = 0.01
    lmbda = 1 / 1000
    epochs = 1000
    x, y, data = get_data(heart_data, features, target_column)
    y = convert_y(y)
    x = normalize(data[train_features])
    x_train, y_train, x_test, y_test = train_test_split(x, y, 0.8)
    w, b = svm(x_train, y_train, learning_rate, lmbda, epochs)
    print(test(x_test, y_test, w, b)/x_test.shape[0])
    # best = -1
    # bestf = []
    # for i in range(1, 2 ** len(features)):
    #     curr = i
    #     idx = 0
    #     f = []
    #     if i % 10 == 0:
    #         print("Iter: " + str(i))
    #         print("Best: " + str(best))
    #         print("BestF: " + str(bestf))
    #
    #     while (curr):
    #         if curr & 1 == 1:
    #             f.append(features[idx])
    #         idx += 1
    #         curr >>= 1
    #
    #     w, b = svm(data[f], y, learning_rate, lmbda, epochs)
    #
    #     acc = test(data[f], y, w, b)
    #     if acc > best:
    #         best = acc
    #         bestf = f

    # print(best)
    # print(bestf)
    # print(w)
    # print(b)
    # print(x.shape)


if __name__ == "__main__":
    main()
