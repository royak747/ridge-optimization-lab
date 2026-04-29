import numpy as np

def ridge_closed_form_solution(X, y, lambda_val):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape
    A = (X.T @ X) / n_samples + lambda_val * np.eye(n_features)
    b = (X.T @ y) / n_samples
    return np.linalg.solve(A, b)

# Cost function: min_w (1/n)||Xw - y||^2 + λ||w||^2,  λ > 0
# Gradient (2/n) X^T(Xw - y) + 2λw

# --- Gradient Descent function for ridge regression
def gradient_descent_ridge(X, y, learning_rate, lambda_val, n_iterations=1000):
        
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)

    w_star = ridge_closed_form_solution(X, y, lambda_val)

    weight_history = [weights.copy()]
    loss_history = []
    distance_to_optimum = []

    for iteration in range(n_iterations + 1):
        predictions = X @ weights
        errors = predictions - y

        loss = np.mean(errors ** 2) + lambda_val * np.sum(weights ** 2)

        loss_history.append(loss)
        distance_to_optimum.append(np.linalg.norm(weights - w_star))

        if iteration == n_iterations:
            break

        gradient = (2 / n_samples) * (X.T @ errors) + 2 * lambda_val * weights
        weights = weights - learning_rate * gradient

        weight_history.append(weights.copy())

    return weights, weight_history, loss_history, distance_to_optimum