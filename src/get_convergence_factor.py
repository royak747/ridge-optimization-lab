import numpy as np

# Helper function to calculate theoretical convergence factor 'c' for GD/Ridge GD
def theoretical_convergence_factor(X, lambda_val, learning_rate):
    n_samples, n_features = X.shape
    # Hessian for Ridge Regression objective: (2/n)X^T X + 2*lambda*I
    H = (2 / n_samples) * (X.T @ X) + 2 * lambda_val * np.eye(n_features)

    # Eigenvalues of the Hessian
    eigenvalues = np.linalg.eigvalsh(H)
    L = eigenvalues.max()  # Largest eigenvalue (Lipschitz constant)
    mu = eigenvalues.min() # Smallest eigenvalue

    # The actual convergence factor for a given learning rate
    c = max(abs(1 - learning_rate * L), abs(1 - learning_rate * mu))
    return c