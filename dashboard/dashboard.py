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

# session state
if 'customer_score' not in st.session_state:
    st.session_state['customer_score'] = None

# Selection of the customer
customer_id = st.selectbox(
    label="Choose the customer_id",
    options=['XXXXXX', *valid_customer_ids],
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
    
    
    
    ####################################################################
    # Main section divided in 2 columns    
    ####################################################################
    st.divider()
    st.write("### Model decision")
    # Compute the customer_score only if unknown.
    if st.session_state.customer_score is None:
        with st.spinner('Wait for the model decision...'):
            # Send request to the model API 
            customer_score, model_threshold = get_customer_score(customer_data) 
        
    # Build and fill the columns   
    main_col1, main_col2 = st.columns([1, 1])    
    with main_col1:
        # Print the model decision
        if customer_score < model_threshold:
            st.write("Loan accepted")
        else:
            st.write("Loan rejected")
            
    with main_col2:
        # place for the gauge
        st.image(
            'https://www.tibco.com/sites/tibco/files/media_entity/2022-01/GaugeChart-01.svg',
            use_column_width=True
        )
                    
    st.divider()
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
    with st.expander("### Compare a customer feature to others"):
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
    with st.sidebar:
        st.write("# Customer data")
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
        # print the data
        if sel_fts == []:
            st.write(customer_data)
        else:
            st.write(customer_data.loc[sel_fts])
        
        # st.button('Reset the feature selection', on_click=clear_multisel)
        
    