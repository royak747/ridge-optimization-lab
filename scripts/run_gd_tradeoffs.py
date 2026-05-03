import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from src.sgd_ridge import ridge_sgd
from src.gradient_descent_ridge import gradient_descent_ridge, ridge_closed_form_solution
from src.standard_gradient_descent import gradient_descent
from src.config import (
    N_ITERATIONS,
    N_EPOCHS_SGD,
    LEARNING_RATE,
    DATA_DIR,
    OUTPUT_DIR,
    LAMBDA_VAL_COMPARISON,
    SELECTED_KAPPA,
    SELECTED_NOISE_STD,
    BATCH_SIZE_SGD
)

# Function for safe handling of values for semilogy plots
# Avoids issues with log(0) or tiny negative values
def safe_semilogy_values(values, eps=1e-14):
    values = np.asarray(values, dtype=float)
    return np.maximum(values, eps)

# Compute ||w_k - w_star|| for every stored iteration in weight_history
def compute_distance_history(weight_history, w_star):
    return np.asarray([np.linalg.norm(np.asarray(w) - w_star) for w in weight_history])

# Sample SGD distance history at the end of each epoch for work-normalized comparison
def sample_sgd_at_epoch_end(distance_sgd, n_epochs, updates_per_epoch):
    epoch_x = []
    epoch_distances = []

    # For each epoch
    # Find corresponding index in distance_sgd for the end of that epoch
    for epoch in range(1, n_epochs + 1):
        idx = epoch * updates_per_epoch
        if idx >= len(distance_sgd):
            break
        # Store the epoch number and corresponding distance for plotting
        epoch_x.append(epoch)
        epoch_distances.append(distance_sgd[idx])

    return np.asarray(epoch_x), np.asarray(epoch_distances)


# Main method where data is loaded so that comparisons can be made
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "datasets.pkl"), "rb") as f:
        datasets_dict = pickle.load(f)

    selected_kappa = SELECTED_KAPPA
    selected_noise_std = SELECTED_NOISE_STD

    # Get selected dataset based on Config values
    data = datasets_dict[(selected_kappa, selected_noise_std)]
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    beta_true = np.asarray(data["beta_true"], dtype=float)

    # Hyperparameters from Config
    lambda_val = LAMBDA_VAL_COMPARISON
    learning_rate = LEARNING_RATE
    n_iterations_gd = N_ITERATIONS
    n_epochs_sgd = N_EPOCHS_SGD
    batch_size_sgd = BATCH_SIZE_SGD

    n_samples = X.shape[0]

    # --- Ridge Gradient Descent
    _, _, loss_history_gd, distance_gd = gradient_descent_ridge(
        X,
        y,
        learning_rate=learning_rate,
        lambda_val=lambda_val,
        n_iterations=n_iterations_gd,
    )
    distance_gd = np.asarray(distance_gd, dtype=float)

    print(
        f"Ridge GD Final Loss: {loss_history_gd[-1]:.4e}, "
        f"Final Distance to Ridge Optimum: {distance_gd[-1]:.4e}"
    )

    # --- Ridge Stochastic Gradient Descent
    _, weight_history_sgd, loss_history_sgd = ridge_sgd(
        X,
        y,
        learning_rate=learning_rate,
        lambda_val=lambda_val,
        n_epochs=n_epochs_sgd,
        batch_size=batch_size_sgd,
    )

    # Closed form solution is used here to compute distance history for SGD
    w_star_ridge = ridge_closed_form_solution(X, y, lambda_val)
    distance_sgd = compute_distance_history(weight_history_sgd, w_star_ridge)

    print(
        f"Ridge SGD Final Loss: {loss_history_sgd[-1]:.4e}, "
        f"Final Distance to Ridge Optimum: {distance_sgd[-1]:.4e}"
    )

    # --- Standard Gradient Descent
    _, weight_history_standard_gd, loss_history_standard_gd, _, parameter_error_history_standard_gd, = gradient_descent(X, 
                                                                                                                        y,
                                                                                                                        learning_rate=learning_rate,
                                                                                                                        n_iterations=n_iterations_gd,
                                                                                                                        beta_true=beta_true,
                                                                                                                        )

    # Using empirical OLS optimum for distance-to-optimum comparison
    # Not using beta_true here, because noisy data means beta_true != empirical optimum
    w_star_standard = np.linalg.lstsq(X, y, rcond=None)[0]
    distance_standard_gd = compute_distance_history(weight_history_standard_gd, w_star_standard)

    print(
        f"Standard GD Final Loss: {loss_history_standard_gd[-1]:.4e}, "
        f"Final Distance to OLS Optimum: {distance_standard_gd[-1]:.4e}, "
        f"Final Parameter Error to beta_true: {parameter_error_history_standard_gd[-1]:.4e}"
    )

       
    # --- GD vs. SGD vs. Standard GD Objective Value Comparison
    max_epoch_like = min(n_epochs_sgd + 1, len(loss_history_gd), len(loss_history_standard_gd))

    plt.figure(figsize=(10, 6))

    plt.semilogy(
        np.arange(max_epoch_like),
        safe_semilogy_values(loss_history_gd[:max_epoch_like]),
        label="Ridge GD Objective",
    )
    plt.semilogy(
        np.arange(1, len(loss_history_sgd) + 1),
        safe_semilogy_values(loss_history_sgd),
        label="Ridge SGD Objective (per Epoch)",
    )
    plt.semilogy(
        np.arange(max_epoch_like),
        safe_semilogy_values(loss_history_standard_gd[:max_epoch_like]),
        label="Standard GD / OLS Objective",
        linestyle="--",
    )

    plt.xlabel("GD Iterations / SGD Epochs")
    plt.ylabel("Objective Value (Log Scale)")
    plt.title(
        "Objective Value Comparison: GD vs. SGD vs. Standard GD "
        f"(kappa={selected_kappa:.0e}, noise={selected_noise_std}, lambda={lambda_val})"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "4. gd_methods_objective_comparison.jpg"), dpi=300)
    plt.show()
    plt.close()


    # --- Work-normalized distance comparison by data passes
    gd_data_passes = np.arange(len(distance_gd), dtype=float)
    standard_data_passes = np.arange(len(distance_standard_gd), dtype=float)
    sgd_updates = np.arange(len(distance_sgd), dtype=float)
    sgd_data_passes = sgd_updates * batch_size_sgd / n_samples

    max_passes = min(float(n_iterations_gd), float(n_epochs_sgd))

    plt.figure(figsize=(10, 6))

    # Plot distance to optimum for all methods on the same graph with semilogy scale
    plt.semilogy(
        gd_data_passes,
        safe_semilogy_values(distance_gd),
        label="Ridge GD Distance to Ridge Optimum",
    )

    # Sample SGD distance at the end of each epoch for work-normalized comparison
    plt.semilogy(
        sgd_data_passes,
        safe_semilogy_values(distance_sgd),
        label="Ridge SGD Distance to Ridge Optimum (work-normalized)",
        alpha=0.8,
    )

    # Plot standard GD distance to OLS optimum for comparison
    plt.semilogy(
        standard_data_passes,
        safe_semilogy_values(distance_standard_gd),
        label="Standard GD Distance to OLS Optimum",
        linestyle="--",
    )

    plt.xlim(0, max_passes)
    plt.xlabel("Approximate Data Passes")
    plt.ylabel("Distance to Corresponding Empirical Optimum (Log Scale)")
    plt.title(
        "Work-Normalized Distance-to-Optimum Comparison "
        f"(kappa={selected_kappa:.0e}, noise={selected_noise_std}, lambda={lambda_val})"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "5. gd_methods_distance_to_optimum_work_normalized.jpg"), dpi=300)
    plt.show()
    plt.close()

# Run command: python -m scripts.run_gd_tradeoffs
if __name__ == "__main__":
    main()
