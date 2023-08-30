import sys
sys.path.append('..')
from project_tools.scorer import Scorer
from project_tools import utils
from sklearn.metrics import roc_auc_score, fbeta_score

# Define Scorers here :
Scorer('AUC', roc_auc_score)
Scorer('f2', fbeta_score, score_kwargs={'beta': 2})
# custom metric : loss_of_income
Scorer('loss_of_income', utils.loss_of_income_score, greater_is_better=False)
## Weighted geometric mean
# Scorer(
#     'recall_specificity_G_mean',
#     weighted_geometric_mean_score,
# ),

