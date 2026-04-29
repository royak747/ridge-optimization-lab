# Reproducibility
SEED = 42

# Hyperparameters for optimization algos
N_ITERATIONS = 20000
LEARNING_RATE = 0.1
N_EPOCHS_SGD = 1000
BATCH_SIZE_SGD = 32

# Dataset generation params
CONDITION_NUMBERS = [10, 1e2, 1e4, 1e6]
NOISE_LEVELS = [0.01, 0.1, 1.0]
N_SAMPLES = 500
N_FEATURES = 20
SELECTED_KAPPA = 1e4
SELECTED_NOISE_STD = 0.1

# Regularization params
LAMBDA_VALS = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
LAMBDA_VAL_COMPARISON = 1e-3

# Paths
DATA_DIR = "data"
OUTPUT_DIR = "artifacts"