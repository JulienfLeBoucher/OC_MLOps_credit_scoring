# Business problem analysis and scoring metric to tune classifiers

## Reminder
In the dataset, when the TARGET column indicates:
- 0 -- the client had no problem to repay the credit.
- 1 -- the client had difficulties to repay.

## Impacts of error types regarding *Home Credit* interests

- The worst case scenario for *Home Credit* would be to authorize a credit to someone who is not to repay it. This happens when the model wrongly predicts the client as : $$healthy\equiv 0 \equiv Negative$$ Hence, any raise of the number of False Negatives $FN$ (Type II error) must be strongly penalized when scoring the model.

- To a lesser extent, overclassifying clients as :$$unhealthy\equiv 1\equiv Positive$$ would also imply a loss of income. *Home Credit* would not leverage the opportunities of making profit in safe configurations. This happens when then number of False Positives $FP$ arises. Thus, we want to minimize it, but it is less dramatic to commit that type I error than the previous one.

## Assumption 
Without any expert advice, let further consider that *the expected loss of a FN is 10 times higher than for FP*. How could we take that into consideration in order to build 




   