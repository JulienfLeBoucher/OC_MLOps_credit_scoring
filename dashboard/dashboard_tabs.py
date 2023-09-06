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
    Get the data about of all customers and cache it.
    """
    df = pd.read_pickle(DATA_URI)
    target = df.pop('TARGET')
    return df, target

@st.cache_data
def get_customer_score(customer_data):
    time.sleep(3)
    return 0.27, 0.16    

def clear_multisel():
    # Callback to clear the multiselect multisel in the sidebar
    # thanks to the button
    st.session_state.multisel = []
    return None


###### Global variable definitions
df, target = get_data()
valid_customer_ids = list(df.index)
fts_by_model_importance = df.columns
# Instantiation to avoid errors.
customer_data = None

with st.sidebar:
    # Selection of the customer
    customer_id = st.selectbox(
        label="Choose the customer_id",
        options=['XXXXXX', *valid_customer_ids],
        key="id"
    )
    if customer_id not in valid_customer_ids:
        st.error("Please select a customer_id.")
    else:
        # Get customer data and format it.
        customer_data = (
            df
            .loc[customer_id, :]
            .T
        )
        customer_data.index.name = 'Feature name'
        customer_data.name = 'Value'
        # Eventually use a pd styler.
        
        st.divider()
        with st.spinner('Wait for the model decision...'):
            # Send request to the model API 
            customer_score, model_threshold = get_customer_score(customer_data) 
        # print the result
        st.write("### Model decision")
        # Print the model decision
        if customer_score < model_threshold:
            st.write("Loan accepted")
        else:
            st.write("Loan rejected")
            
        # Print gauge to tell if close from the decision boundary
        # place for the gauge
        st.write("### Location relative to the decision boundary")
        st.image(
            'https://www.tibco.com/sites/tibco/files/media_entity/2022-01/GaugeChart-01.svg',
            use_column_width=True
        )
            
        
# start tabs       
tab1, tab2, tab3 = st.tabs(
    [
        "Main factors in the decision  ",
        "Explore customer's data  ",
        "Compare customer to others",
    ]
)

with tab1:
    if customer_id == 'XXXXXX':
        st.write('')
    else:
    ####################################################################
    # Middle section : Observe customer major factors to the decision.
    ####################################################################
        st.write("## Major factors in the decision")
        
        st.image("https://i.stack.imgur.com/Ftxu7.png")
        
        st.write(
            "A value pushing the decision to the left helps in getting"
            " a loan. Otherwise, it plays against the customer."
        )    
####################################################################
# Bottom section : compare data to others
####################################################################
with tab2:
    ft = st.selectbox(
        label="",
        options=fts_by_model_importance,
        placeholder="Choose the feature"
    )
    # if not ft:
    #     st.error("Please select a feature.")
    ft_values = df.loc[:, ft]    

####################################################################
# Left side bar
####################################################################
# Display the whole customer data or only selected features 
with tab3:
    col1, _, col2 = st.columns([1, 0.2, 1])
    with col2:
        st.write(
            "If no features are selected, you can scroll down the table "
            "to explore the whole data of the customer.\n\n"
            "Otherwise, you are invited to select multiple features with "
            "the selector right below.\n\n"
            "*Note* : In the selectors, Features are order by their "
            "importance in the model decision."
        )
        # Feature selector        
        sel_fts = st.multiselect(
            label='',
            options=fts_by_model_importance,
            default=None,
            key='multisel',
            placeholder="Select features here.",
            label_visibility='hidden',
        )
    with col1:    
        # print the data
        if sel_fts == []:
            st.write(customer_data)
        else:
            st.write(customer_data.loc[sel_fts])
    
    # st.button('Reset the feature selection', on_click=clear_multisel)
    
