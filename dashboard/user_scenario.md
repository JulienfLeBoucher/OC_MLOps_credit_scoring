# Dashboard

## Users
The dashboard users are customer relationship workers at *"prêts à dépenser"*.

## Utility
The dashboard exists to help in:
- announcing the customer the model decision (loan or not).
- stating if the customer was on the decision boundary and thus could improve
his status easily to change the decision or not.
- enumerating the major factors to that decision.
- exploring one customer's data in detail and comparing it with a similar group of clients/all clients.

## Components to implement

- client ID
- model decision field
- probability gauge with the decision threshold and the client position.
- shap waterfall for the top 10 factors.
- A text zone to select a feature which would return :
    - the value we know about the customer.
    - the global distribution of the feature and the client value in it
    - the distribution of similar customers and the value in it.
    - maybe the comparison between distribution of risky customers and safe customers so that one can interpret if the value needs to be increased or decreased to improve the prediction.

