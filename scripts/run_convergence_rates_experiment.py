"""Run convergence-rate experiments for OLS GD, Ridge GD, and Ridge SGD.

This script fixes several issues in the earlier version:
  * OLS/standard GD is run with `gradient_descent`, not `gradient_descent_ridge(lambda=0)`.
  * The OLS optimum is computed with `np.linalg.lstsq`, which is safer than solving
    the normal equations when lambda=0.
  * Logs are protected against zero / non-finite distances.
  * Empirical slopes are fit only on the finite tail of the trajectory, rather than
    over the entire transient.
  * Theoretical lines are anchored at the first plotted log-distance, not at an
    empirical-fit intercept.
  * The SGD epoch indexing uses `BATCH_SIZE_SGD` instead of a hard-coded 32.
  * The output directory is created before saving plots.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


EPS = 1e-14


def safe_ridge_solution(X, y, lambda_val):
    """Return the OLS/ridge optimum for the objective used in the project.

    Objective: (1/n)||Xw - y||^2 + lambda_val ||w||^2.

    For lambda=0, use lstsq instead of solving the normal equations directly.
    This is more stable when X^T X is singular or nearly singular.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if lambda_val == 0:
        return np.linalg.lstsq(X, y, rcond=None)[0]

    return ridge_closed_form_solution(X, y, lambda_val)


def theoretical_convergence_factor(X, lambda_val, learning_rate):
    """Spectral-radius convergence factor for full-batch GD on the quadratic."""
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape

    # Hessian of (1/n)||Xw-y||^2 + lambda ||w||^2.
    H = (2.0 / n_samples) * (X.T @ X) + 2.0 * lambda_val * np.eye(n_features)
    eigenvalues = np.linalg.eigvalsh(H)

    L = float(eigenvalues.max())
    mu = float(eigenvalues.min())
    factor = max(abs(1.0 - learning_rate * L), abs(1.0 - learning_rate * mu))

    return factor, mu, L


def log_distances(distances, eps=EPS):
    """Convert distances to finite log-distances."""
    distances = np.asarray(distances, dtype=float)
    distances = np.where(np.isfinite(distances), distances, np.nan)
    distances = np.maximum(distances, eps)
    return np.log(distances)


def fit_empirical_slope(x, y, start_frac=0.2, end_frac=1.0):
    """Fit a line to the finite tail of a log-distance curve."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if len(x) < 2:
        return np.nan, np.nan

    start = int(np.floor(start_frac * len(x)))
    end = int(np.ceil(end_frac * len(x)))
    start = min(max(start, 0), len(x) - 2)
    end = min(max(end, start + 2), len(x))

    slope, intercept = np.polyfit(x[start:end], y[start:end], 1)
    return float(slope), float(intercept)


def theoretical_line(iterations, log_distance, theoretical_slope):
    """Plot theoretical slope anchored at the first empirical point."""
    iterations = np.asarray(iterations, dtype=float)
    log_distance = np.asarray(log_distance, dtype=float)

    finite_mask = np.isfinite(iterations) & np.isfinite(log_distance)
    if not np.any(finite_mask) or not np.isfinite(theoretical_slope):
        return np.full_like(iterations, np.nan, dtype=float)

    first_idx = np.flatnonzero(finite_mask)[0]
    x0 = iterations[first_idx]
    y0 = log_distance[first_idx]
    return y0 + theoretical_slope * (iterations - x0)


def warn_if_unstable(name, factor, learning_rate, L):
    """Print a simple stability diagnostic for full-batch GD."""
    print(f"{name}: theoretical factor={factor:.6f}, slope={np.log(factor):.6e}")
    print(f"{name}: L={L:.6e}, stable step-size upper bound ≈ {2.0 / L:.6e}")

    if factor >= 1:
        print(
            f"WARNING: {name} has factor >= 1 with learning_rate={learning_rate}. "
            "Full-batch GD is not predicted to contract in the worst case."
        )


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

    # Centering y makes the no-intercept model reasonable.
    y_mean = y_train.mean()
    y_train = y_train - y_mean
    y_test = y_test - y_mean

    lambda_val = LAMBDA_VAL_COMPARISON
    learning_rate = LEARNING_RATE
    n_iterations_gd = N_ITERATIONS
    n_epochs_sgd = N_EPOCHS_SGD
    batch_size_sgd = BATCH_SIZE_SGD

    print("Running convergence-rate experiment")
    print(f"learning_rate={learning_rate}")
    print(f"lambda={lambda_val}")
    print(f"n_iterations_gd={n_iterations_gd}")
    print(f"n_epochs_sgd={n_epochs_sgd}")
    print(f"batch_size_sgd={batch_size_sgd}")

    # Optima
    w_star_std = safe_ridge_solution(X_train, y_train, lambda_val=0)
    w_star_ridge = safe_ridge_solution(X_train, y_train, lambda_val=lambda_val)

    # Standard GD
    weights_std, weight_history_std, loss_std, grad_norm_std, _ = gradient_descent(
        X_train,
        y_train,
        learning_rate=learning_rate,
        n_iterations=n_iterations_gd,
    )
    distance_std = np.array([np.linalg.norm(w - w_star_std) for w in weight_history_std])

    # Ridge GD
    weights_gd, weight_history_gd, loss_gd, distance_gd = gradient_descent_ridge(
        X_train,
        y_train,
        learning_rate=learning_rate,
        lambda_val=lambda_val,
        n_iterations=n_iterations_gd,
    )
    distance_gd = np.asarray(distance_gd, dtype=float)

    # Ridge SGD
    weights_sgd, weight_history_sgd, loss_sgd = ridge_sgd(
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
    c_ols, mu_ols, L_ols = theoretical_convergence_factor(
        X_train, lambda_val=0, learning_rate=learning_rate
    )
    c_ridge, mu_ridge, L_ridge = theoretical_convergence_factor(
        X_train, lambda_val=lambda_val, learning_rate=learning_rate
    )

    warn_if_unstable("OLS GD", c_ols, learning_rate, L_ols)
    warn_if_unstable("Ridge GD", c_ridge, learning_rate, L_ridge)

    theoretical_slope_ols = np.log(c_ols) if c_ols > 0 else -np.inf
    theoretical_slope_ridge = np.log(c_ridge) if c_ridge > 0 else -np.inf

    # Logs and empirical slopes
    iterations_std = np.arange(len(distance_std))
    iterations_gd = np.arange(len(distance_gd))
    iterations_sgd = np.arange(len(distance_sgd))

    log_distance_std = log_distances(distance_std)
    log_distance_gd = log_distances(distance_gd)
    log_distance_sgd = log_distances(distance_sgd)

    empirical_slope_std, empirical_intercept_std = fit_empirical_slope(
        iterations_std, log_distance_std
    )
    empirical_slope_gd, empirical_intercept_gd = fit_empirical_slope(
        iterations_gd, log_distance_gd
    )
    empirical_slope_sgd, empirical_intercept_sgd = fit_empirical_slope(
        iterations_sgd, log_distance_sgd
    )

    print(f"Empirical slope (Standard GD): {empirical_slope_std:.6e}")
    print(f"Empirical slope (Ridge GD):    {empirical_slope_gd:.6e}")
    print(f"Empirical slope (Ridge SGD):   {empirical_slope_sgd:.6e} per mini-batch update")

    # SGD sampled at end of each epoch. weight_history_sgd includes the initial weight
    # at index 0, then one entry after each mini-batch update.
    updates_per_epoch = int(np.ceil(X_train.shape[0] / batch_size_sgd))
    epoch_end_indices = np.arange(1, n_epochs_sgd + 1) * updates_per_epoch
    epoch_end_indices = epoch_end_indices[epoch_end_indices < len(distance_sgd)]

    sgd_epochs_x_axis = np.arange(1, len(epoch_end_indices) + 1)
    log_distance_sgd_per_epoch = log_distance_sgd[epoch_end_indices]

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
    plt.savefig(
        os.path.join(OUTPUT_DIR, "7. convergence_rate_comparison_gd_ridge_gd.jpg"),
        dpi=300,
        bbox_inches="tight",
    )
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
    plt.savefig(
        os.path.join(OUTPUT_DIR, "8. convergence_rate_ridge_sgd_per_update.jpg"),
        dpi=300,
        bbox_inches="tight",
    )
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
    plt.savefig(
        os.path.join(OUTPUT_DIR, "9. convergence_rate_ridge_sgd_per_epoch.jpg"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved plots to:")
    print(os.path.join(OUTPUT_DIR, "7. convergence_rate_comparison_gd_ridge_gd.jpg"))
    print(os.path.join(OUTPUT_DIR, "8. convergence_rate_ridge_sgd_per_update.jpg"))
    print(os.path.join(OUTPUT_DIR, "9. convergence_rate_ridge_sgd_per_epoch.jpg"))


if __name__ == "__main__":
    main()
