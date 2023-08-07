import os

# PATHS
PROJECT_DIRECTORY = "/home/louberehc/OCR/projets/7_scoring_model"
FEATURE_PATH = os.path.join(
    PROJECT_DIRECTORY,
    "pickle_files/features.pkl.gz"
)

# LightGBM model - configurations and hyperparameters
NUM_THREADS = 4
NUM_FOLDS = 5
RANDOM_SEED = 737851

