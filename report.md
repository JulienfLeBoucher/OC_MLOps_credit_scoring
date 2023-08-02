# Business problem analysis

## Reminder
In the dataset, when the TARGET column indicates:
- 0 -- the client had no problem to repay the credit.
- 1 -- the client had difficulties to repay.

## Impacts of error types regarding *Home Credit* interests

- The worst case scenario for *Home Credit* would be to authorize a credit to someone who is not to repay it. This happens when the model wrongly predicts the client as : $$healthy\equiv 0 \equiv Negative$$ Hence, the False Negatives (Type II error) must be minimized prioritarily. $$FNR = 1-TPR = 1-recall$$
This is equivalent to **maximize the recall very prioritarily**.

- To a lesser extent, overclassifying clients as unhealthy would imply a loss of income because *Home Credit* would allow less contract than what is safely possible, resulting in less opportunities to make profit. This happens when the False Positive Rate $FPR$ is too high. Thus, we want to minimize it, but it is less dramatic to commit that type I error than the previous one. 
$$FPR=1 - TNR=1-specifity 
   