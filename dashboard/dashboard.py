import streamlit as st
import pandas as pd
import time

DATA_URI = (
    '/home/louberehc/OCR/projets/7_scoring_model/'
    'pickle_files/features_sample.pkl'
)

@st.cache_data
def get_data():
    """
    Get the data about of all customers
    """
    df = pd.read_pickle(DATA_URI)
    target = df.pop('TARGET')
    return df, target


def get_model_prediction(customer_data):
    return st.write('test_model_prediction')    

######
df, target = get_data()

customer_id = st.multiselect(
    label="Choose the customer_id",
    options=list(df.index),
)

if not customer_id:
    st.error("Please select a customer_id.")
elif len(customer_id) > 1:
    st.error("Please make sure to select only one customer_id.")
else:
    customer_data = df.loc[customer_id, :]
    customer_target = target.loc[customer_id]
    st.divider()
    st.write('### Customer data', customer_data)
    # st.write('class :', customer_target)
    if st.button('Model decision'):
        with st.spinner('Wait for it...'):
            time.sleep(3)
        # st.success('Done!')gress
        st.write(
            'make model prediction and display cutoff result and '
            'decision boundary gauge HERE.')
    st.divider()
    st.write('### Compare customer data to others')
    
    fts = list(df.columns)
    ft = st.multiselect(
        label="Choose one feature",
        options=fts,
    )
    if not ft:
        st.error("Please select a feature.")
    elif len(ft) > 1:
        st.error("Please make sure to select only one feature.")
    
    ft_values = df.loc[:, ft]    
    
    
    