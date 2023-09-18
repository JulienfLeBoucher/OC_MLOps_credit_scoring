import my_app
import api_utils
import unittest
import numpy as np

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

class MyTestCase(unittest.TestCase):
        
    def test_customer_and_shap_values_correspondence(self):
        """
        Test if the get_index() function enables to extract the 
        shap values related to the right customer.
        
        This is true when the sum of the shap values + base value
        are related to the probability predicted by the model.
        
        Here, equality is tested with a relative tolerance.
        
        For robustness, it is tested on several customers.
        """
        rtol = 1e-3
        # Pick three customers.
        customer_ids = [400991, 191921, 353368]
        # Prepare vector to store and compare.
        model_proba = np.zeros(len(customer_ids))
        shap_proba = np.zeros_like(model_proba)
        for n, customer_id in enumerate(customer_ids):
            # Get the customer probability predicted by the model
            model_proba[n] = api_utils.get_customer_proba(
                my_app.model, my_app.features, customer_id
            )
            # Get the index (row number of the shap values to retrieve)
            idx = api_utils.get_index(customer_id, my_app.features)
            # Get the customer shap explanation according that index
            customer_exp = my_app.exp[idx]
            # Compute the sum of the shap values on the log-odds scale
            shap_sum = customer_exp.base_values + np.sum(customer_exp.values)
            # Convert to probability
            shap_proba[n] = inv_logit(shap_sum)
        # Check both vectors are almost equal.    
        self.assertTrue(
            np.allclose(
                model_proba,
                shap_proba,
                rtol=rtol,
            )
        )

    def test_if_prediction_route_print_valid_customers(self):
        """
        Test that some of the customer_ids provided by the
        '/prediction' route are in `valid_customer_ids`.
        """
        customer_ids = [400991, 191921, 353368]
        self.assertTrue(
            set(my_app.valid_customer_ids).issuperset(set(customer_ids))
        )