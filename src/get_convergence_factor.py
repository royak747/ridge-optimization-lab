import numpy as np

# Helper function to calculate theoretical convergence factor 'c' for GD/Ridge GD
def theoretical_convergence_factor(X, lambda_val, learning_rate):
    # Spectral-radius convergence factor for full-batch GD on the quadratic objective
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape

    # Hessian of (1/n)||Xw-y||^2 + lambda ||w||^2.
    H = (2.0 / n_samples) * (X.T @ X) + 2.0 * lambda_val * np.eye(n_features)
    eigenvalues = np.linalg.eigvalsh(H)

    # L is the largest eigenvalue, mu is the smallest
    L = float(eigenvalues.max())
    mu = float(eigenvalues.min())
    # Connvergence factor
    # Denoted as p
    # p = max({|1 - learning_rate * L|, |1 - learning_rate * mu|})
    factor = max(abs(1.0 - learning_rate * L), abs(1.0 - learning_rate * mu))
    return factor, mu, L