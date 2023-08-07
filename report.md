# Business problem analysis and scoring metric to tune classifiers

## Reminder
In the dataset, when the TARGET column indicates:
- 0 -- the client had no problem to repay the credit.
- 1 -- the client had difficulties to repay.

## Impacts of error types regarding *Home Credit* interests

- The worst case scenario for *Home Credit* would be to authorize a credit to someone who is not to repay it (even partially). This happens when the model wrongly predicts the client as : $$healthy\equiv 0 \equiv Negative$$ Hence, any False Negative (Type II error) must be strongly penalized when scoring the model.

- To a lesser extent, overclassifying clients as :$$unhealthy\equiv 1\equiv Positive$$ would also imply a loss of income. *Home Credit* would not leverage the opportunities of making profit in safe configurations. This happens with a False Positive. Thus, we also want to minimize them, but it is less dramatic to make that type I error than the previous one.

## Assumption 
Without any expert advice, let further consider that *the expected financial loss of a False Negative is 5 times higher than for a False Positive*. How could we take that into consideration to find the model maximizing *Home Credit* profits?

## Scoring functions that emphasize on errors

Among the classical metrics to score classifiers, some penalize False Positives and False Negatives.
### Recall
$$recall =\dfrac{TP}{TP+FN}$$
This formula suggests intuitively that the less false negatives, the better the recall. 

### Precision

$$precision =\dfrac{TP}{TP+FP}$$
Intuitively, the less false positives, the the better the precision. 
  
### $F_\beta$ score function

While one would love to be able to max both previous metrics, some trade-off exists because of there interactivity.

Fortunately, we can find a balance focusing harder on optimizing the recall than precision with the $F_\beta$ score function which is a weighted harmonic mean of $recall$ and $precision$.

Some maths exposed [here](https://stats.stackexchange.com/questions/503511/precision-and-recall-for-highly-imbalanced-data) demonstrate the following formula :
$$F_\beta=\dfrac{(1+\beta)\times TP}{(1+\beta)\times TP + \beta \times FN + FP }$$





   