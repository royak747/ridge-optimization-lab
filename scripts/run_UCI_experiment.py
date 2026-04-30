import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from src.sgd_ridge import ridge_sgd
from src.gradient_descent_ridge import gradient_descent_ridge, ridge_closed_form_solution
from src.standard_gradient_descent import gradient_descent
from src.UCI_data_gen import load_ucirepo_dataset
from src.config import (
    N_ITERATIONS,
    N_EPOCHS_SGD,
    LEARNING_RATE,
    DATA_DIR,
    OUTPUT_DIR,
    LAMBDA_VAL_COMPARISON,
    SELECTED_KAPPA,
    SELECTED_NOISE_STD,
    BATCH_SIZE_SGD,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    X, y = load_ucirepo_dataset("boston_housing")

    # Convert pandas objects to numpy
    X_real = X.to_numpy()

    # Choose one target: heating load
    y_real = y.iloc[:, 0].to_numpy()   # Y1

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

    # Standardize features
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    # Center target
    y_mean = y_train.mean()
    y_train = y_train - y_mean

    # --- Plot Convergence Behavior using Ridge GD
    lambda_vals = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    learning_rate = 0.1

    plt.figure(figsize=(8, 5))

    for lambda_val in lambda_vals:
        weights, weight_history, loss_history, distance_history = gradient_descent_ridge(
            X_train,
            y_train,
            learning_rate=learning_rate,
            lambda_val=lambda_val,
            n_iterations=2000
        )

        plt.semilogy(loss_history, label=fr"lambda={lambda_val}")

    plt.xlabel("Iteration")
    plt.ylabel("Ridge Objective")
    plt.title("UCI Energy Efficiency: Ridge GD Objective vs. Iteration")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "uci_ridge_gd_loss_vs_iterations.jpg"))

    # --- Re-run the optimization to capture distance to optimum histories
    lambda_val_comparison = 1e-4
    n_iterations_gd = 2000
    n_epochs_sgd = 2000

    # Standard GD: lambda = 0
    _, _, loss_std, distance_std = gradient_descent_ridge(
        X_train, y_train,
        learning_rate=0.01,
        lambda_val=0,
        n_iterations=n_iterations_gd
    )

    # Ridge GD
    _, _, loss_gd, distance_gd = gradient_descent_ridge(
        X_train, y_train,
        learning_rate=0.01,
        lambda_val=lambda_val_comparison,
        n_iterations=n_iterations_gd
    )

    # Ridge SGD (using ridge_sgd_with_history for full weight history)
    _, weight_history_sgd_real, loss_sgd = ridge_sgd(
        X_train, y_train,
        learning_rate=0.01,
        lambda_val=lambda_val_comparison,
        n_epochs=n_epochs_sgd,
        batch_size=32
    )

    # Calculate optimal weights for comparison
    w_star_std = ridge_closed_form_solution(X_train, y_train, 0) # OLS solution
    w_star_ridge = ridge_closed_form_solution(X_train, y_train, lambda_val_comparison)

    # Recalculate distance to optimum for SGD with the actual w_star_ridge
    distance_sgd_real = [np.linalg.norm(w - w_star_ridge) for w in weight_history_sgd_real]

    plt.figure(figsize=(10, 6))

    plt.semilogy(distance_std, label='Standard GD Distance to Optimum', linestyle='--')
    plt.semilogy(distance_gd, label=f'Ridge GD Distance to Optimum, λ={lambda_val_comparison}')

    # Plot SGD Distance to Optimum, aligning per epoch
    updates_per_epoch_real = int(np.ceil(X_train.shape[0] / 32)) # Assuming batch_size=32
    sgd_epochs_x_axis_real = np.arange(1, n_epochs_sgd + 1)
    sgd_distances_at_epoch_end_real = [distance_sgd_real[i * updates_per_epoch_real] for i in sgd_epochs_x_axis_real]
    plt.semilogy(sgd_epochs_x_axis_real, sgd_distances_at_epoch_end_real, label=f'Ridge SGD Distance to Optimum (per Epoch), λ={lambda_val_comparison}')

    plt.xlabel("Iterations / Epochs")
    plt.ylabel("||w - w*|| (Log Scale)")
    plt.title("UCI Energy Efficiency: GD vs. Ridge GD vs. Ridge SGD (Distance to Optimum)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "uci_gd_sgd_distance_to_optimum_comparison.jpg"))


# Run command: python -m scripts.run_UCI_experiment
if __name__ == "__main__":
    main()