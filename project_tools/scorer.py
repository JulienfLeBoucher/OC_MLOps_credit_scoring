import numpy as np

class Scorer:
    """ 
    Scorer which can:
    
    - evaluate predictions when given y_true and y_pred as classes.
    - find the best threshold to optimize the score when given 
    probabilities and compute the best score.
    
    """
    # to track all instances
    all = []
    
    def __init__(
        self,
        name: str,
        score_func,
        score_kwargs: dict()=None, 
        greater_is_better: bool=True,
    ):
        # Attributes
        self.name = name
        self.score_func = score_func
        self.score_kwargs = score_kwargs
        self.greater_is_better = greater_is_better
        # Track instances
        Scorer.all.append(self)

    def evaluate_predictions(self, y_true, y_pred):
        """ Compute the score."""
        if self.score_kwargs:
            return self.score_func(y_true, y_pred, **self.score_kwargs)
        else:
            return self.score_func(y_true, y_pred)
            
    def find_best_threshold_and_score(
        self,
        y_true,
        proba_pred,
        threshold_step=0.01
    ):
        eps = threshold_step * 0.01
        thresholds = np.arange(0, 1+eps, threshold_step)
        scores = np.zeros_like(thresholds)
        for idx, t in enumerate(thresholds):
            scores[idx] = self.evaluate_predictions(
                y_true,
                threshold_class_from_proba(proba_pred, threshold=t)    
            )
        if self.greater_is_better:
            best_score = np.max(scores)
            best_threshold = thresholds[np.argmax(scores)]  
        else:
            best_score = np.min(scores)
            best_threshold = thresholds[np.argmin(scores)]  
        return best_threshold, best_score
    
    def print(self):
        print(f"name : {self.name}")
        print(f"score_func : {self.score_func}")
        print(f"score_kwargs : {self.score_kwargs}")
        print(f"greater_is_better : {self.greater_is_better}")
# End Scorer