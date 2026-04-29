import numpy as np

# Cost function: min_w 1/n||Xw - y||^2
# Gradient: (2/n)(X^T dot (Xw - y))

# --- Gradient Descent function
def gradient_descent(X, y, learning_rate, n_iterations=1000, random_state=None, beta_true=None):
    
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if beta_true is not None:
        beta_true = np.asarray(beta_true, dtype=float)

    n_samples, n_features = X.shape

    # Initialize weights
    weights = np.zeros(n_features)

    # Store optimization history
    weight_history = [weights.copy()]
    loss_history = [np.mean((X @ weights - y) ** 2)]
    gradient_norm_history = []
    parameter_error_history = []

    if beta_true is not None:
        parameter_error_history.append(np.linalg.norm(weights - beta_true))

    for iteration in range(n_iterations):
        # Current predictions and errors
        predictions = X @ weights
        errors = predictions - y

        # Gradient of (1/n)||Xw - y||^2
        gradient = (2 / n_samples) * (X.T @ errors)

        # Store gradient norm
        gradient_norm_history.append(np.linalg.norm(gradient))

        # Gradient descent update
        weights = weights - learning_rate * gradient

        # Store updated weights
        weight_history.append(weights.copy())

        # Compute loss at updated weights
        updated_predictions = X @ weights
        updated_errors = updated_predictions - y
        loss_history.append(np.mean(updated_errors ** 2))

        if beta_true is not None:
            parameter_error_history.append(np.linalg.norm(weights - beta_true))

    return weights, weight_history, loss_history, gradient_norm_history, parameter_error_history

