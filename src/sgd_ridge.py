import numpy as np

# --- Mini-Batch Stochastic Gradient Descent function for ridge regression
def ridge_sgd(X, y, learning_rate, lambda_val, n_epochs=100, batch_size=32, random_state=42):
    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)

    # Store optimization history
    loss_history = []
    weight_history = [weights.copy()]

    for epoch in range(n_epochs):
        # Shuffle data at beginning of each epoch
        indices = rng.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # For each mini-batch, compute gradient and update wights
        for start in range(0, n_samples, batch_size):
            # Get current mini-batch
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Batch size used for this mini-batch
            batch_size_actual = X_batch.shape[0]

            # Compute errors
            errors = X_batch @ weights - y_batch

            # Gradient of ridge regression for mini-batch
            gradient = (2 / batch_size_actual) * (X_batch.T @ errors) + 2 * lambda_val * weights

            # Update weights
            weights -= learning_rate * gradient
            weight_history.append(weights.copy()) # Store weights after each update

        # Store loss after each epoch
        full_errors = X @ weights - y
        loss = np.mean(full_errors ** 2) + lambda_val * np.sum(weights ** 2)
        loss_history.append(loss)

    return weights, weight_history, loss_history