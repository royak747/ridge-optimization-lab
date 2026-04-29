import numpy as np
import os
import json
from src.data_generation import generate_ill_conditioned_data
from src.config import CONDITION_NUMBERS, NOISE_LEVELS, DATA_DIR

def main():
    datasets = {}

    # Generate datasets for each noise level for every condition number
    # In this case, we will end up with 12 datasets of varying conditioning and noise
    for kappa in CONDITION_NUMBERS:
        for noise_std in NOISE_LEVELS:
            X, y, beta_true, singular_values = generate_ill_conditioned_data(
                n_samples=500,
                n_features=20,
                condition_number=kappa,
                noise_std=noise_std,
                random_state=42
            )

            datasets[(kappa, noise_std)] = {
                "X": X.tolist(),
                "y": y.tolist(),
                "beta_true": beta_true.tolist(),
                "singular_values": singular_values.tolist(),
                "actual_condition_number": float(np.linalg.cond(X))
            }
            
            print(
                f"kappa target={kappa:.0e}, "
                f"noise={noise_std}, "
                f"actual cond(X)={np.linalg.cond(X):.2e}, "
                f"cond(X^T X)={np.linalg.cond(X.T @ X):.2e}"
            )

            # Save datasets to disk for later use as json file
            with open(os.path.join(DATA_DIR, f"dataset_kappa_{kappa}_noise_{noise_std}.json"), "w") as f:
                json.dump(datasets[(kappa, noise_std)], f)

# Run command: python -m scripts.run_data_generation
if __name__ == "__main__":
    main()