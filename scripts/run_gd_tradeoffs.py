import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

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

    print(
        f"Comparing methods for kappa={selected_kappa:.0e}, "
        f"noise={selected_noise_std}, lambda={lambda_val}"
    )

    # Ridge Gradient Descent
    weights_gd, weight_history_gd, loss_history_gd, distance_gd = gradient_descent_ridge(
        X,
        y,
        learning_rate=learning_rate,
        lambda_val=lambda_val,
        n_iterations=n_iterations_gd,
    )

    print(
        f"Ridge GD Final Loss: {loss_history_gd[-1]:.4e}, "
        f"Final Distance to Ridge Optimum: {distance_gd[-1]:.4e}"
    )


    # Ridge Stochastic Gradient Descent
    weights_sgd, weight_history_sgd, loss_history_sgd = ridge_sgd(
        X,
        y,
        learning_rate=learning_rate,
        lambda_val=lambda_val,
        n_epochs=n_epochs_sgd,
        batch_size=batch_size_sgd,
    )

    w_star = ridge_closed_form_solution(X, y, lambda_val)
    distance_sgd = [np.linalg.norm(w - w_star) for w in weight_history_sgd]

    print(
        f"Ridge SGD Final Loss: {loss_history_sgd[-1]:.4e}, "
        f"Final Distance to Ridge Optimum: {distance_sgd[-1]:.4e}"
    )

    # Standard Gradient Descent, no ridge penalty
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

    print(
        f"Standard GD Final Loss: {loss_history_standard_gd[-1]:.4e}, "
        f"Final Parameter Error: {parameter_error_history_standard_gd[-1]:.4e}"
    )

    # Compute closed-form solution for standard (non-ridge) least squares
    w_star_standard = np.linalg.pinv(X) @ y

    # Compute distance to optimum over iterations
    distance_standard_gd = [
        np.linalg.norm(w - w_star_standard)
        for w in weight_history_standard_gd
    ]

    # Plot objective value comparison
    plt.figure(figsize=(10, 6))

    plt.semilogy(loss_history_gd[: n_epochs_sgd + 1], label="GD Ridge Objective")
    plt.semilogy(
        np.arange(1, len(loss_history_sgd) + 1),
        loss_history_sgd,
        label="SGD Ridge Objective (per Epoch)",
    )
    plt.semilogy(
        loss_history_standard_gd[: n_epochs_sgd + 1],
        label="Standard GD Objective",
        linestyle="--",
    )

    plt.xlabel("Iterations / Epochs")
    plt.ylabel("Objective Value (Log Scale)")
    plt.title(
        "Comparison of GD vs SGD vs Standard GD Ridge: Objective Value "
        f"(kappa={selected_kappa:.0e}, noise={selected_noise_std}, lambda={lambda_val})"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gd_methods_objective_comparison.jpg"))

    # Plot distance-to-optimum comparison
    updates_per_epoch = int(np.ceil(X.shape[0] / batch_size_sgd))
    sgd_epochs_x_axis = np.arange(1, n_epochs_sgd + 1)

    sgd_distances_at_epoch_end = [
        distance_sgd[i * updates_per_epoch]
        for i in sgd_epochs_x_axis
    ]

    plt.figure(figsize=(10, 6))

    plt.semilogy(
        distance_gd[: n_epochs_sgd + 1],
        label="GD Ridge Distance to Optimum",
    )

    plt.semilogy(
        sgd_epochs_x_axis,
        sgd_distances_at_epoch_end,
        label="SGD Ridge Distance to Optimum (per Epoch)",
    )

    plt.semilogy(
        parameter_error_history_standard_gd[: n_epochs_sgd + 1],
        label="Standard GD Parameter Error",
        linestyle="--",
    )

    plt.xlabel("Iterations / Epochs")
    plt.ylabel("||w - w*|| (Log Scale)")
    plt.title(
        "Comparison of GD vs SGD vs Standard GD Ridge: Distance to Optimum "
        f"(kappa={selected_kappa:.0e}, noise={selected_noise_std}, lambda={lambda_val})"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gd_methods_distance_to_optimum.jpg"))


# Run command: python -m scripts.run_gd_tradeoffs
if __name__ == "__main__":
    main()
    
'''
These plots illustrate the comparative convergence behavior of gradient descent (GD), stochastic gradient descent (SGD), and ridge-regularized gradient descent under an ill-conditioned linear regression setting. In the objective value plot, SGD exhibits the fastest decrease in the loss function, rapidly converging to a lower objective value within relatively few epochs. This behavior is expected, as stochastic updates allow SGD to make more frequent progress and can help it navigate poorly conditioned directions more effectively than full-batch methods. In contrast, ridge gradient descent converges more slowly, which is consistent with the effect of ill-conditioning and the added regularization term. Standard gradient descent, while unregularized, shows slightly faster convergence than ridge GD in terms of objective value, likely because it does not incur the additional penalty term and therefore optimizes a less constrained objective.

The distance-to-optimum plot further highlights differences in convergence dynamics. SGD again demonstrates the most rapid decrease, approaching the ridge-optimal solution much faster than full-batch ridge GD. Ridge GD shows a gradual but steady reduction in distance to its optimum, reflecting the slower convergence typical of gradient descent in ill-conditioned problems. The standard gradient descent curve, however, remains relatively flat because it is plotted using the parameter error with respect to the true underlying coefficient vector rather than the least-squares optimum. Due to the presence of noise in the data, the true parameter vector does not coincide with the empirical optimum, and as a result, the parameter error does not approach zero even as the algorithm converges.

Overall, the results demonstrate that SGD is more efficient in practice for ill-conditioned problems, achieving faster convergence both in terms of objective value and proximity to the optimum. Ridge regularization improves numerical stability but can slow convergence, while standard gradient descent without regularization may converge more quickly in objective value but does not necessarily recover the true parameters in the presence of noise.'''