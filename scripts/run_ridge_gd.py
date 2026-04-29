import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from src.gradient_descent_ridge import gradient_descent_ridge
from src.config import CONDITION_NUMBERS, NOISE_LEVELS, N_ITERATIONS, LEARNING_RATE, DATA_DIR, LAMBDA_VALS, LAMBDA_VAL_COMPARISON, SELECTED_KAPPA, SELECTED_NOISE_STD

def main():
    lambda_vals = LAMBDA_VALS
    with open(os.path.join(DATA_DIR, "datasets.pkl"), "rb") as f:
        datasets_dict = pickle.load(f)

    X_selected = datasets_dict[(SELECTED_KAPPA, SELECTED_NOISE_STD)]["X"]
    y_selected = datasets_dict[(SELECTED_KAPPA, SELECTED_NOISE_STD)]["y"]

    for lambda_val in lambda_vals:
        #lr = ridge_lr(X, lambda_val)
        lr = 0.1

        weights, weight_history, loss_history, distance_history = gradient_descent_ridge(X_selected, y_selected, learning_rate=lr, lambda_val=lambda_val, n_iterations=1000)

        print(
            f"lambda={lambda_val:.0e}, "
            f"final loss={loss_history[-1]:.4e}, "
            f"distance to optimum={distance_history[-1]:.4e}"
        )

    # Graphing
    plt.figure(figsize=(12, 5))
    for lambda_val in lambda_vals:
        weights, weight_history, loss_history, distance_history = gradient_descent_ridge(
            X_selected, y_selected,
            learning_rate=0.1,
            lambda_val=lambda_val,
            n_iterations=1000
        )
        plt.semilogy(loss_history, label=fr"$\lambda={lambda_val}$")
    plt.xlabel("Iteration")
    plt.ylabel("Ridge Objective")
    plt.title("Ridge Gradient Descent: Objective vs. Iteration")
    plt.legend()
    plt.grid(True)
    # Save the graph to artifacts/
    plt.savefig(os.path.join("artifacts", "ridge_gd_loss_vs_iterations.jpg"))

    # Another graph
    plt.figure(figsize=(10, 6))
    for lambda_val in lambda_vals:
        weights, weight_history, loss_history, distance_history = gradient_descent_ridge(
            X_selected, y_selected,
            learning_rate=0.1,
            lambda_val=lambda_val,
            n_iterations=1000
        )

        plt.semilogy(distance_history, label=f"lambda={lambda_val}")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\|w_k - w^*\|$")
    plt.title("Ridge GD: Distance to Optimum")
    plt.legend()
    plt.grid(True)
    # Save the graph to artifacts/
    plt.savefig(os.path.join("artifacts", "ridge_gd_distance_to_optimum.jpg"))

            

# Run command: python -m scripts.run_ridge_gd
if __name__ == "__main__":
    main()