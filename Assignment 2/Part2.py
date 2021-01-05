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
    return x, y


# WX + b
def function(x, w, b):
    return np.dot(x, w) + b


def predict(x, w, b):
    return -1 if function(x, w, b) < 0 else 1


# Takes y {0,1} and returns y {-1,1}
def convert_y(y):
    return np.where(y < 0, -1, 1)


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
    for _ in range(epochs):
        for i, curr in enumerate(x):
            if y[i] * function(curr, w, b) >= 1:
                w = w - alpha * 2 * lmbda * w
            else:
                w = w + alpha * (np.dot(curr, y[i]) - 2 * lmbda * w)
                b = b - alpha * y[i]
    return x, w, b


def visualize_features(X, features, y):
    plt.scatter(X[features[0]], X[features[1]], marker='o', c=y)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.show()


def main():
    heart_data = 'part2_data/heart.csv'
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                'thal']
    target_column = 'target'

    x, y = get_data(heart_data, features, target_column)

    # visualize different combination to see the best
    # visualize_features(x, ['age', 'trestbps'], y)
    # visualize_features(x, ['age', 'sex'], y)  # some how acceptable
    # visualize_features(x, ['age', 'cp'], y)
    # visualize_features(x, ['age', 'chol'], y)
    # visualize_features(x, ['age', 'chol'], y)
    # visualize_features(x, ['thalach', 'age'], y)
    # visualize_features(x, ['sex', 'oldpeak'], y)
    # visualize_features(x, ['exang', 'oldpeak'], y)



    learning_rate = 0.01
    lmbda = 0.001
    epochs = 200

if __name__ == "__main__":
    main()
