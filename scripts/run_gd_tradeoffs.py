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
    BATCH_SIZE_SGD,
)


def safe_semilogy_values(values, eps=1e-14):
    """Avoid log-scale issues from exact zeros or tiny negative numerical artifacts."""
    values = np.asarray(values, dtype=float)
    return np.maximum(values, eps)


def compute_distance_history(weight_history, w_star):
    """Compute ||w_k - w_star|| for every stored iterate."""
    return np.asarray([np.linalg.norm(np.asarray(w) - w_star) for w in weight_history])


def sample_sgd_at_epoch_end(distance_sgd, n_epochs, updates_per_epoch):
    """
    distance_sgd[0] is the initial point before any mini-batch update.
    End of epoch e occurs after e * updates_per_epoch updates.
    """
    epoch_x = []
    epoch_distances = []

    for epoch in range(1, n_epochs + 1):
        idx = epoch * updates_per_epoch
        if idx >= len(distance_sgd):
            break
        epoch_x.append(epoch)
        epoch_distances.append(distance_sgd[idx])

    return np.asarray(epoch_x), np.asarray(epoch_distances)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "datasets.pkl"), "rb") as f:
        datasets_dict = pickle.load(f)

    selected_kappa = SELECTED_KAPPA
    selected_noise_std = SELECTED_NOISE_STD

    data = datasets_dict[(selected_kappa, selected_noise_std)]
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    beta_true = np.asarray(data["beta_true"], dtype=float)

    lambda_val = LAMBDA_VAL_COMPARISON
    learning_rate = LEARNING_RATE
    n_iterations_gd = N_ITERATIONS
    n_epochs_sgd = N_EPOCHS_SGD
    batch_size_sgd = BATCH_SIZE_SGD

    n_samples = X.shape[0]
    updates_per_epoch = int(np.ceil(n_samples / batch_size_sgd))

    print(
        f"Comparing methods for kappa={selected_kappa:.0e}, "
        f"noise={selected_noise_std}, lambda={lambda_val}, "
        f"learning_rate={learning_rate}, batch_size={batch_size_sgd}"
    )

    # ------------------------------------------------------------
    # Ridge Gradient Descent
    # ------------------------------------------------------------
    weights_gd, weight_history_gd, loss_history_gd, distance_gd = gradient_descent_ridge(
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

    # ------------------------------------------------------------
    # Ridge Stochastic Gradient Descent
    # ------------------------------------------------------------
    weights_sgd, weight_history_sgd, loss_history_sgd = ridge_sgd(
        X,
        y,
        learning_rate=learning_rate,
        lambda_val=lambda_val,
        n_epochs=n_epochs_sgd,
        batch_size=batch_size_sgd,
    )

    w_star_ridge = ridge_closed_form_solution(X, y, lambda_val)
    distance_sgd = compute_distance_history(weight_history_sgd, w_star_ridge)

    print(
        f"Ridge SGD Final Loss: {loss_history_sgd[-1]:.4e}, "
        f"Final Distance to Ridge Optimum: {distance_sgd[-1]:.4e}"
    )

    # ------------------------------------------------------------
    # Standard Gradient Descent / OLS
    # ------------------------------------------------------------
    (
        weights_standard_gd,
        weight_history_standard_gd,
        loss_history_standard_gd,
        gradient_norm_history_standard_gd,
        parameter_error_history_standard_gd,
    ) = gradient_descent(
        X,
        y,
        learning_rate=learning_rate,
        n_iterations=n_iterations_gd,
        beta_true=beta_true,
    )

    # Use empirical OLS optimum for distance-to-optimum comparison.
    # Do not use beta_true here, because noisy data means beta_true != empirical optimum.
    w_star_standard = np.linalg.lstsq(X, y, rcond=None)[0]
    distance_standard_gd = compute_distance_history(weight_history_standard_gd, w_star_standard)

    print(
        f"Standard GD Final Loss: {loss_history_standard_gd[-1]:.4e}, "
        f"Final Distance to OLS Optimum: {distance_standard_gd[-1]:.4e}, "
        f"Final Parameter Error to beta_true: {parameter_error_history_standard_gd[-1]:.4e}"
    )

    # ------------------------------------------------------------
    # Plot 1: Objective value comparison by epoch-like x-axis
    # ------------------------------------------------------------
    # This reproduces the intuitive comparison, but note that one SGD epoch contains
    # several mini-batch updates, while one GD iteration contains one full-gradient update.
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
    plt.savefig(os.path.join(OUTPUT_DIR, "4. gd_methods_objective_comparison.jpg"), dpi=300)
    plt.close()

    # Plot 2: Distance-to-optimum comparison by epoch-like x-axis
    sgd_epochs_x_axis, sgd_distances_at_epoch_end = sample_sgd_at_epoch_end(
        distance_sgd,
        n_epochs_sgd,
        updates_per_epoch,
    )

    max_epoch_like = min(n_epochs_sgd + 1, len(distance_gd), len(distance_standard_gd))

    plt.figure(figsize=(10, 6))

    plt.semilogy(
        np.arange(max_epoch_like),
        safe_semilogy_values(distance_gd[:max_epoch_like]),
        label="Ridge GD Distance to Ridge Optimum",
    )
    plt.semilogy(
        sgd_epochs_x_axis,
        safe_semilogy_values(sgd_distances_at_epoch_end),
        label="Ridge SGD Distance to Ridge Optimum (per Epoch)",
    )
    plt.semilogy(
        np.arange(max_epoch_like),
        safe_semilogy_values(distance_standard_gd[:max_epoch_like]),
        label="Standard GD Distance to OLS Optimum",
        linestyle="--",
    )

    plt.xlabel("GD Iterations / SGD Epochs")
    plt.ylabel("Distance to Corresponding Empirical Optimum (Log Scale)")
    plt.title(
        "Distance-to-Optimum Comparison: GD vs. SGD vs. Standard GD "
        f"(kappa={selected_kappa:.0e}, noise={selected_noise_std}, lambda={lambda_val})"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5. gd_methods_distance_to_optimum.jpg"), dpi=300)
    plt.close()

    # Plot 3: Work-normalized distance comparison by data passes
    # This is the fairest comparison of computational effort:
    #   GD iteration = one full pass over n samples
    #   SGD update = batch_size / n passes over the data
    gd_data_passes = np.arange(len(distance_gd), dtype=float)
    standard_data_passes = np.arange(len(distance_standard_gd), dtype=float)
    sgd_updates = np.arange(len(distance_sgd), dtype=float)
    sgd_data_passes = sgd_updates * batch_size_sgd / n_samples

    max_passes = min(float(n_iterations_gd), float(n_epochs_sgd))

    plt.figure(figsize=(10, 6))

    plt.semilogy(
        gd_data_passes,
        safe_semilogy_values(distance_gd),
        label="Ridge GD Distance to Ridge Optimum",
    )
    plt.semilogy(
        sgd_data_passes,
        safe_semilogy_values(distance_sgd),
        label="Ridge SGD Distance to Ridge Optimum (work-normalized)",
        alpha=0.8,
    )
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
    plt.savefig(os.path.join(OUTPUT_DIR, "6. gd_methods_distance_to_optimum_work_normalized.jpg"), dpi=300)
    plt.close()

'''    print("Saved plots:")
    print(os.path.join(OUTPUT_DIR, "gd_methods_objective_comparison.jpg"))
    print(os.path.join(OUTPUT_DIR, "gd_methods_distance_to_optimum.jpg"))
    print(os.path.join(OUTPUT_DIR, "gd_methods_distance_to_optimum_work_normalized.jpg"))'''


# Run command: python -m scripts.run_gd_tradeoffs
if __name__ == "__main__":
    main()
