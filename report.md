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

## Classical scoring functions 
Among the classical metrics to score classifiers, some penalize False Positives and False Negatives.

### FNR (False negative rate)

We absolutely want to minimize $FNR$.
$$FNR= 1-recall$$
Thus, it is equivalent to maximize the recall.
### Recall or sensitivity
$$recall =\dfrac{TP}{TP+FN}$$
This formula suggests intuitively that the less false negatives, the better the recall. It is consistent. As stated before, I would love to maximize it.

### Precision 

$$precision =\dfrac{TP}{TP+FP}$$
Intuitively, the less false positives, the better the precision. So we also want to maximize that, but with less priority than recall.

  
### $F_\beta$ score function

As a conclusion of previous paragraphs, I ideally would want to max simultaneously recall and precision metrics, but some trade-off exist because of there interactivity.

Fortunately, we can find a balance focusing harder on optimizing the recall than precision with the $F_\beta$ score function which is a weighted harmonic mean of $recall$ and $precision$.

Some maths exposed [here](https://stats.stackexchange.com/questions/503511/precision-and-recall-for-highly-imbalanced-data) demonstrate the following formula :
$$F_\beta=\dfrac{(1+\beta)\times TP}{(1+\beta)\times TP + \beta \times FN + FP }$$

It is not obvious how to chose the right $\beta$ to reflect precisely the assumption made. Indeed, the higher the $\beta$, the less $FP$ will influence because $(1+\beta)\times TP$ will start dominate numerator et denominator. $\beta=5$ would not truly reflect that a $FN$ is 5 times more penalizing than a $FP$.

### Weighted geometric mean
One could also consider balancing between the recall(sensitivity) and the specificity using a weighted geometric mean such as, for example:
$$G_\beta=\sqrt[\beta+1]{recall^{\beta}\times specificity}$$

## A custom metric corresponding directly to the assumption
The two last metrics would probably yield good results, but choosing the value of $\beta$ to make a precise use of the assumption is far from being trivial.

Here is a very simple evaluation metric (to be minimized) we could use.

Let's simply count in the model's predictions the loss of income $LoI$ by counting $FP$ and $FN$ and weighting $FN$ as follow: $$LoI=5FN+FP$$ It has the disadvantage not to be clipped in $[0,1]$, but it does not need to be normalized as all models are trained on the same training set, making it comparable.  





   