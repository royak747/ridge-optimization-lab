import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from src.standard_gradient_descent import gradient_descent
from src.config import CONDITION_NUMBERS, NOISE_LEVELS, N_ITERATIONS, LEARNING_RATE, DATA_DIR

def main():
    n_iterations = N_ITERATIONS # CONFIG
    learning_rate = LEARNING_RATE # CONFIG: A fixed learning rate to observe the effects of ill-conditioning

    gd_results = {}

    print(f"Running Gradient Descent for {n_iterations} iterations with learning rate {learning_rate}...")

    # Get the json files from data/
    with open(os.path.join(DATA_DIR, "datasets.pkl"), "rb") as f:
        datasets = pickle.load(f)

    # Loop through each dataset, load the data, and run standard gradient descent    
    for (kappa, noise_std), data in datasets.items():
        X = data["X"]
        y = data["y"]
        beta_true = data["beta_true"]

        # Run standard GD and store results
        weights_final, weight_history, loss_history, gradient_norm_history, parameter_error_history = gradient_descent(
            X, y, learning_rate=learning_rate, n_iterations=n_iterations, beta_true=beta_true
        )

        gd_results[(kappa, noise_std)] = {
            "weights_final": weights_final,
            "weight_history": weight_history,
            "loss_history": loss_history,
            "gradient_norm_history": gradient_norm_history,
            "parameter_error_history": parameter_error_history
        }

        print(
            f"Completed GD for kappa={kappa:.0e}, noise={noise_std}. " # Print dataset info
            f"Final Loss: {loss_history[-1]:.4f}, Final Parameter Error: {parameter_error_history[-1]:.4f}" # print final loss and parameter error for eaach dataset
        )

    # --- Graphing
    # Plot loss history for each dataset
    plt.figure(figsize=(15, 5 * len(NOISE_LEVELS)))
    plt.suptitle('Objective Value (MSE) vs. Iterations', fontsize=16)

    for i, noise_std in enumerate(NOISE_LEVELS):
        plt.subplot(len(NOISE_LEVELS), 1, i + 1)
        for kappa in CONDITION_NUMBERS:
            key = (kappa, noise_std)
            if key in gd_results:
                history = gd_results[key]["loss_history"]
                plt.plot(history, label=f'kappa={kappa:.0e}', alpha=0.8)
        plt.title(f'Noise Std: {noise_std}')
        plt.xlabel('Iterations')
        plt.ylabel('Objective Value (MSE)')
        plt.yscale('log') # Use log scale for better visualization of decay
        plt.legend(loc='upper right')
        plt.grid(True, which="both", ls="--", c='0.7')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the graph to artifacts/
    plt.savefig(os.path.join("artifacts", "1. gd_loss_vs_iterations.jpg"))
    # plt.show()
        
# Run command: python -m scripts.run_standard_gd
if __name__ == "__main__":
    main()