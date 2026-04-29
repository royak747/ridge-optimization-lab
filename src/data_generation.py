import numpy as np

# Generate a synthetic linear regression dataset with a controlled condition number.
'''
    model:
        y = X beta_true + noise

    The singular values of X are constructed so that:
        condition_number(X) = largest singular value / smallest singular value
'''
def generate_ill_conditioned_data(n_samples=500, n_features=2, condition_number=1e4, noise_std=0.1, random_state=42):
    rng = np.random.default_rng(random_state)

    # Random orthonormal matrices U and V
    A = rng.normal(size=(n_samples, n_features))
    U, _ = np.linalg.qr(A)

    B = rng.normal(size=(n_features, n_features))
    V, _ = np.linalg.qr(B)

    # Singular values decay from 1 to 1 / condition_number
    singular_values = np.geomspace(1, 1 / condition_number, n_features)

    # Construct X = U S V^T
    X = U @ np.diag(singular_values) @ V.T

    # True parameter vector
    beta_true = rng.normal(size=n_features)

    # Gaussian noise
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_samples)

    # Response variable
    y = X @ beta_true + noise

    return X, y, beta_true, singular_values