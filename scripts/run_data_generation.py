import numpy as np
import os
import json
import pickle
from src.data_generation import generate_ill_conditioned_data
from src.config import CONDITION_NUMBERS, NOISE_LEVELS, DATA_DIR, SEED, N_SAMPLES, N_FEATURES

def main():
    datasets = {}

    # Generate datasets for each noise level for every condition number
    # In this case, we will end up with 12 datasets of varying conditioning and noise
    for kappa in CONDITION_NUMBERS:
        for noise_std in NOISE_LEVELS:
            X, y, beta_true, singular_values = generate_ill_conditioned_data(
                n_samples=N_SAMPLES,
                n_features=N_FEATURES,
                condition_number=kappa,
                noise_std=noise_std,
                random_state=SEED
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
            # Save it as a dictionary where I can access as datasets.items()
            with open(os.path.join(DATA_DIR, "datasets.pkl"), "wb") as f:
                pickle.dump(datasets, f)


# Run command: python -m scripts.run_data_generation
if __name__ == "__main__":
    main()