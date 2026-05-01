import numpy as np

# Closed form ridge:
# w* = (X^T X / n + λI)^(-1) X^T y / n
def ridge_closed_form_solution(X, y, lambda_val):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape
    A = (X.T @ X) / n_samples + lambda_val * np.eye(n_features)
    b = (X.T @ y) / n_samples
    return np.linalg.solve(A, b)

# --- Gradient Descent function for ridge regression
# Cost function: min_w (1/n)||Xw - y||^2 + λ||w||^2,  λ > 0
# Gradient (2/n) X^T(Xw - y) + 2λw
def gradient_descent_ridge(X, y, learning_rate, lambda_val, n_iterations=1000):
        
    # Get X, y as numpy arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape

    # Initialize weights
    weights = np.zeros(n_features)

    # Compute closed-form solution to track distance to optimum later
    w_star = ridge_closed_form_solution(X, y, lambda_val)

    # Store optimization history
    weight_history = [weights.copy()]
    loss_history = []
    distance_to_optimum = []

    # For each iteration, compute gradient and update weights
    for iteration in range(n_iterations + 1):
        predictions = X @ weights
        errors = predictions - y

        # Objective function value at current weights
        loss = np.mean(errors ** 2) + lambda_val * np.sum(weights ** 2)

        loss_history.append(loss)
        distance_to_optimum.append(np.linalg.norm(weights - w_star))

        # Gradient for ridge regression
        gradient = (2 / n_samples) * (X.T @ errors) + 2 * lambda_val * weights
        weights = weights - learning_rate * gradient

        weight_history.append(weights.copy())

    return weights, weight_history, loss_history, distance_to_optimum