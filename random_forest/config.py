import os

MAIN_DIR = "/home/louberehc/OCR/projets/7_scoring_model"
DATA_PATH = os.path.join(MAIN_DIR, 'pickle_files/features.pkl.gz')
TRACKING_URI = os.path.join(MAIN_DIR, 'hyperparameter_tuning/mlruns')


NUM_FOLDS = 5
RANDOM_SEED = 137
