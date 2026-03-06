import math

def mean(data):
    return sum(data) / len(data)

def variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)

def standard_deviation(data):
    return math.sqrt(variance(data))

def z_score(data):
    m = mean(data)
    std = standard_deviation(data)
    return [(x - m) / std for x in data]

def dot_product(x, y):
    return sum(x[i] * y[i] for i in range(len(x)))

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    n = len(y_true)
    loss = 0
    for i in range(n):
        loss += y_true[i] * math.log(y_pred[i]) + \
                (1 - y_true[i]) * math.log(1 - y_pred[i])
    return -loss / n


def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)

def precision(y_true, y_pred):
    tp = 0
    fp = 0
    for i in range(len(y_true)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                tp += 1
            else:
                fp += 1
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                tp += 1
            else:
                fn += 1
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    return 2 * (p * r) / (p + r)

def linear_regression_gradient_descent(X, Y, lr=0.01, epochs=1000):
    n = len(X)
    w = 0
    b = 0

    for _ in range(epochs):
        dw = 0
        db = 0

        for i in range(n):
            y_pred = w * X[i] + b
            error = y_pred - Y[i]

            dw += error * X[i]
            db += error

        dw = dw / n
        db = db / n

        w -= lr * dw
        b -= lr * db

    return w, b


if __name__ == "__main__":

    data = [1, 2, 3, 4, 5]

    print("Mean:", mean(data))
    print("Variance:", variance(data))
    print("Standard Deviation:", standard_deviation(data))
    print("Z-Scores:", z_score(data))

    x_vec = [1, 2, 3]
    y_vec = [4, 5, 6]
    print("Dot Product:", dot_product(x_vec, y_vec))

    print("Sigmoid(0):", sigmoid(0))

    X = [1, 2, 3, 4, 5]
    Y = [2, 4, 6, 8, 10]

    w, b = linear_regression_gradient_descent(X, Y)
    print("Trained Weight:", w)
    print("Trained Bias:", b)


    y_true_prob = [0, 1, 1, 0]
    y_pred_prob = [0.1, 0.9, 0.8, 0.2]
    print("Binary Cross Entropy Loss:", binary_cross_entropy(y_true_prob, y_pred_prob))


    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]

    print("Accuracy:", accuracy(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("Recall:", recall(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))