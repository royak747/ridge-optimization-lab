import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

from src.config import (
    BATCH_SIZE_SGD,
    DATA_DIR,
    LAMBDA_VAL_COMPARISON,
    LEARNING_RATE,
    N_EPOCHS_SGD,
    N_ITERATIONS,
    OUTPUT_DIR,
    SELECTED_KAPPA,
    SELECTED_NOISE_STD,
)

from src.gradient_descent_ridge import gradient_descent_ridge, ridge_closed_form_solution
from src.sgd_ridge import ridge_sgd
from src.standard_gradient_descent import gradient_descent
from src.get_convergence_factor import theoretical_convergence_factor


# Numerical stability threshold for log-distance plots
EPS = 1e-14


# Objective: (1/n)||Xw - y||^2 + lambda_val ||w||^2.
def safe_ridge_solution(X, y, lambda_val):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # If lambda = 0, least squares solution computed
    # Using np.linalg.lstsq
    if lambda_val == 0:
        return np.linalg.lstsq(X, y, rcond=None)[0]

    return ridge_closed_form_solution(X, y, lambda_val)


# Helper functions for plotting and slope fitting
def log_distances(distances, eps=EPS):
    # Convert distances to finite log-distances
    distances = np.asarray(distances, dtype=float)
    distances = np.where(np.isfinite(distances), distances, np.nan)
    distances = np.maximum(distances, eps)
    return np.log(distances)


# Fit a line to the finite tail of a log-distance curve, returning slope and intercept
def fit_empirical_slope(x, y, start_frac=0.2, end_frac=1.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Filter to finite points only for fitting
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    # Check if there are at least 2 points
    if len(x) < 2:
        return np.nan, np.nan

    # Fit a line to the tail of the curve, starting from start_frac to end_frac of the finite points
    start = int(np.floor(start_frac * len(x)))
    end = int(np.ceil(end_frac * len(x)))
    start = min(max(start, 0), len(x) - 2)
    end = min(max(end, start + 2), len(x))
    slope, intercept = np.polyfit(x[start:end], y[start:end], 1)
    return float(slope), float(intercept)


# Plot a line with a given slope anchored at the first finite point of the log-distance curve
def theoretical_line(iterations, log_distance, theoretical_slope):
    # Plot theoretical slope anchored at the first empirical point
    iterations = np.asarray(iterations, dtype=float)
    log_distance = np.asarray(log_distance, dtype=float)

    # Check if there are any finite points to anchor the theoretical line
    finite_mask = np.isfinite(iterations) & np.isfinite(log_distance)
    if not np.any(finite_mask) or not np.isfinite(theoretical_slope):
        return np.full_like(iterations, np.nan, dtype=float)

    # Anchor the theoretical line at the first finite point of the log-distance curve
    first_idx = np.flatnonzero(finite_mask)[0]
    x0 = iterations[first_idx]
    y0 = log_distance[first_idx]
    return y0 + theoretical_slope * (iterations - x0)


# FEW NOTES:
# Empirical slopes are fit only on the finite tail of the trajectory, rather than over the entire transient
# Theoretical lines are anchored at the first plotted log-distance, not at an empirical-fit intercept
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data loading and preprocessing
    # Load from data/datasets.pkl, which should contain a single dataset with the selected kappa and noise std.
    with open(os.path.join(DATA_DIR, "datasets.pkl"), "rb") as f:
        data_dict = pickle.load(f)

    data = data_dict[(SELECTED_KAPPA, SELECTED_NOISE_STD)]

    X = data["X"]
    y = data["y"]

    # Convert safely regardless of list / array
    X_real = np.asarray(X, dtype=float)
    y_real = np.asarray(y, dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X_real,
        y_real,
        test_size=0.2,
        random_state=42,
    )

    # Centering y
    y_mean = y_train.mean()
    y_train = y_train - y_mean
    y_test = y_test - y_mean

    # Experiment parameters from Config
    lambda_val = LAMBDA_VAL_COMPARISON
    learning_rate = LEARNING_RATE
    n_iterations_gd = N_ITERATIONS
    n_epochs_sgd = N_EPOCHS_SGD
    batch_size_sgd = BATCH_SIZE_SGD

    # Optima
    w_star_std = safe_ridge_solution(X_train, y_train, lambda_val=0)
    w_star_ridge = safe_ridge_solution(X_train, y_train, lambda_val=lambda_val)

    # Standard GD
    _, weight_history_std, _, _, _ = gradient_descent(
        X_train,
        y_train,
        learning_rate=learning_rate,
        n_iterations=n_iterations_gd,
    )
    distance_std = np.array([np.linalg.norm(w - w_star_std) for w in weight_history_std])

    # Ridge GD
    _, _, _, distance_gd = gradient_descent_ridge(
        X_train,
        y_train,
        learning_rate=learning_rate,
        lambda_val=lambda_val,
        n_iterations=n_iterations_gd,
    )
    distance_gd = np.asarray(distance_gd, dtype=float)

    # Ridge SGD
    _, weight_history_sgd, _ = ridge_sgd(
        X_train,
        y_train,
        learning_rate=learning_rate,
        lambda_val=lambda_val,
        n_epochs=n_epochs_sgd,
        batch_size=batch_size_sgd,
        random_state=42,
    )
    distance_sgd = np.array([np.linalg.norm(w - w_star_ridge) for w in weight_history_sgd])


    # Theoretical full-batch GD slopes
    c_ols, _, _ = theoretical_convergence_factor(X_train, lambda_val=0, learning_rate=learning_rate)
    c_ridge, _, _ = theoretical_convergence_factor(X_train, lambda_val=lambda_val, learning_rate=learning_rate)

    # Convert convergence factors to theoretical slopes for log-distance plots
    theoretical_slope_ols = np.log(c_ols) if c_ols > 0 else -np.inf
    theoretical_slope_ridge = np.log(c_ridge) if c_ridge > 0 else -np.inf

    # Logs and empirical slopes
    iterations_std = np.arange(len(distance_std))
    iterations_gd = np.arange(len(distance_gd))
    iterations_sgd = np.arange(len(distance_sgd))

    log_distance_std = log_distances(distance_std)
    log_distance_gd = log_distances(distance_gd)
    log_distance_sgd = log_distances(distance_sgd)

    # The following fits a line to the tail of the log-distance curve for each method
    empirical_slope_std, empirical_intercept_std = fit_empirical_slope(iterations_std, log_distance_std)
    empirical_slope_gd, empirical_intercept_gd = fit_empirical_slope(iterations_gd, log_distance_gd)
    empirical_slope_sgd, empirical_intercept_sgd = fit_empirical_slope(iterations_sgd, log_distance_sgd)

    print(f"Empirical slope (Standard GD): {empirical_slope_std:.6e}")
    print(f"Empirical slope (Ridge GD):    {empirical_slope_gd:.6e}")
    print(f"Empirical slope (Ridge SGD):   {empirical_slope_sgd:.6e} per mini-batch update")

    # SGD sampled at end of each epoch
    # weight_history_sgd includes the initial weight
    # at index 0, then one entry after each mini-batch update.
    updates_per_epoch = int(np.ceil(X_train.shape[0] / batch_size_sgd))
    epoch_end_indices = np.arange(1, n_epochs_sgd + 1) * updates_per_epoch
    epoch_end_indices = epoch_end_indices[epoch_end_indices < len(distance_sgd)]

    sgd_epochs_x_axis = np.arange(1, len(epoch_end_indices) + 1)
    log_distance_sgd_per_epoch = log_distance_sgd[epoch_end_indices]

    # Fit empirical slope for the epoch-sampled SGD curve
    empirical_slope_sgd_per_epoch, empirical_intercept_sgd_per_epoch = fit_empirical_slope(
        sgd_epochs_x_axis,
        log_distance_sgd_per_epoch,
    )
    print(f"Empirical slope (Ridge SGD):   {empirical_slope_sgd_per_epoch:.6e} per epoch")

    # Plot 1: OLS GD vs Ridge GD
    plt.figure(figsize=(12, 7))

    plt.plot(
        iterations_std,
        log_distance_std,
        label="Standard GD / OLS (Empirical)",
        color="red",
        alpha=0.6,
    )
    plt.plot(
        iterations_std,
        empirical_slope_std * iterations_std + empirical_intercept_std,
        "--",
        color="red",
        label=f"Standard GD / OLS Empirical Fit, slope={empirical_slope_std:.4e}",
    )
    plt.plot(
        iterations_std,
        theoretical_line(iterations_std, log_distance_std, theoretical_slope_ols),
        ":",
        color="red",
        label=f"Standard GD / OLS Theoretical Slope={theoretical_slope_ols:.4e}",
    )

    plt.plot(
        iterations_gd,
        log_distance_gd,
        label=f"Ridge GD (Empirical, λ={lambda_val})",
        color="blue",
        alpha=0.6,
    )
    plt.plot(
        iterations_gd,
        empirical_slope_gd * iterations_gd + empirical_intercept_gd,
        "--",
        color="blue",
        label=f"Ridge GD Empirical Fit, slope={empirical_slope_gd:.4e}",
    )
    plt.plot(
        iterations_gd,
        theoretical_line(iterations_gd, log_distance_gd, theoretical_slope_ridge),
        ":",
        color="blue",
        label=f"Ridge GD Theoretical Slope={theoretical_slope_ridge:.4e}",
    )

    plt.xlabel("Iterations")
    plt.ylabel("Log(||w - w*||)")
    plt.title("Convergence Rate Comparison: GD vs. Ridge GD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6. convergence_rate_comparison_gd_ridge_gd.jpg"))
    plt.close()

    # Plot 2: Ridge SGD by mini-batch update
    plt.figure(figsize=(10, 6))
    plt.plot(
        iterations_sgd,
        log_distance_sgd,
        label=f"Ridge SGD (Empirical, λ={lambda_val})",
        color="green",
        alpha=0.6,
    )
    plt.plot(
        iterations_sgd,
        empirical_slope_sgd * iterations_sgd + empirical_intercept_sgd,
        "--",
        color="green",
        label=f"Ridge SGD Empirical Fit, slope={empirical_slope_sgd:.4e}",
    )
    plt.xlabel("Mini-batch Updates")
    plt.ylabel("Log(||w - w*||)")
    plt.title("Ridge SGD Empirical Convergence Rate by Mini-batch Update")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "7. convergence_rate_ridge_sgd_per_update.jpg"))
    plt.close()

    # Plot 3: Ridge SGD by epoch
    plt.figure(figsize=(10, 6))
    plt.plot(
        sgd_epochs_x_axis,
        log_distance_sgd_per_epoch,
        label=f"Ridge SGD (Empirical, λ={lambda_val})",
        color="purple",
        alpha=0.7,
    )
    plt.plot(
        sgd_epochs_x_axis,
        empirical_slope_sgd_per_epoch * sgd_epochs_x_axis + empirical_intercept_sgd_per_epoch,
        "--",
        color="purple",
        label=f"Ridge SGD Empirical Fit per Epoch, slope={empirical_slope_sgd_per_epoch:.4e}",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Log(||w - w*||)")
    plt.title("Ridge SGD Empirical Convergence Rate by Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "8. convergence_rate_ridge_sgd_per_epoch.jpg"))
    plt.show()
    plt.close()


# Run command: python -m scripts.run_convergence_rates_experiment
if __name__ == "__main__":
    main()
