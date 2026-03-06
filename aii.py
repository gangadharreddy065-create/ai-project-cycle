# ==========================================
# MACHINE LEARNING MATH & METRICS FROM SCRATCH
# No external libraries
# ==========================================

import math

# ------------------------------------------
# 1. VECTOR OPERATIONS
# ------------------------------------------
def vector_add(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]

def vector_subtract(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]

def dot_product(v1, v2):
    return sum(v1[i] * v2[i] for i in range(len(v1)))

def scalar_multiply(scalar, v):
    return [scalar * x for x in v]

# ------------------------------------------
# 2. STATISTICS
# ------------------------------------------
def mean(data):
    return sum(data) / len(data)

def variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)

def standard_deviation(data):
    return math.sqrt(variance(data))

def covariance(x, y):
    mean_x = mean(x)
    mean_y = mean(y)
    return sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / len(x)

def z_score(data):
    m = mean(data)
    sd = standard_deviation(data)
    return [(x - m) / sd for x in data]

# ------------------------------------------
# 3. LINEAR REGRESSION (Closed Form)
# ------------------------------------------
def linear_regression(x, y):
    m = covariance(x, y) / variance(x)
    b = mean(y) - m * mean(x)
    return m, b

def predict_linear(x, m, b):
    return [m * xi + b for xi in x]

# ------------------------------------------
# 4. SIGMOID FUNCTION
# ------------------------------------------
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# ------------------------------------------
# 5. BINARY CROSS ENTROPY LOSS
# ------------------------------------------
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = max(min(y_pred, 1 - epsilon), epsilon)
    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))

# ------------------------------------------
# 6. GRADIENT DESCENT (Linear Regression)
# ------------------------------------------
def gradient_descent(x, y, learning_rate=0.01, epochs=1000):
    m = 0
    b = 0
    n = len(x)
    for _ in range(epochs):
        y_pred = [m * x[i] + b for i in range(n)]
        dm = (-2/n) * sum(x[i] * (y[i] - y_pred[i]) for i in range(n))
        db = (-2/n) * sum(y[i] - y_pred[i] for i in range(n))
        m -= learning_rate * dm
        b -= learning_rate * db
    return m, b

# ------------------------------------------
# 7. MATRIX MULTIPLICATION
# ------------------------------------------
def matrix_multiply(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            sum_value = 0
            for k in range(len(B)):
                sum_value += A[i][k] * B[k][j]
            row.append(sum_value)
        result.append(row)
    return result

# ------------------------------------------
# 8. CLASSIFICATION METRICS
# ------------------------------------------
def accuracy_score(y_true, y_pred):
    correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return correct / len(y_true)

def precision_score(y_true, y_pred):
    tp = sum(1 for i in range(len(y_true)) if y_true[i]==1 and y_pred[i]==1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i]==0 and y_pred[i]==1)
    return tp / (tp + fp) if (tp + fp) != 0 else 0

def recall_score(y_true, y_pred):
    tp = sum(1 for i in range(len(y_true)) if y_true[i]==1 and y_pred[i]==1)
    fn = sum(1 for i in range(len(y_true)) if y_true[i]==1 and y_pred[i]==0)
    return tp / (tp + fn) if (tp + fn) != 0 else 0

def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) != 0 else 0

# ==========================================
# TEST SECTION
# ==========================================
if __name__ == "__main__":

    # Sample data
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]

    print("Mean:", mean(x))
    print("Variance:", variance(x))
    print("Standard Deviation:", standard_deviation(x))
    print("Z-Score:", z_score(x))

    m, b = linear_regression(x, y)
    print("\nClosed Form Linear Regression")
    print("Slope:", m)
    print("Intercept:", b)
    print("Predictions:", predict_linear(x, m, b))

    m_gd, b_gd = gradient_descent(x, y)
    print("\nGradient Descent Result")
    print("Slope:", m_gd)
    print("Intercept:", b_gd)

    print("\nSigmoid(2):", sigmoid(2))
    print("Binary Cross Entropy Example:", binary_cross_entropy(1, 0.9))

    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    print("\nMatrix Multiplication:", matrix_multiply(A, B))

    # Classification metrics example
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]

    print("\nClassification Metrics")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))