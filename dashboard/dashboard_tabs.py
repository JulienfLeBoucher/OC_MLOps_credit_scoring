import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time


DATA_URI = (
    '/home/louberehc/OCR/projets/7_scoring_model/'
    'pickle_files/features_sample.pkl'
)

def ChangeWidgetFontSize(wgt_txt, wch_font_size = '12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.fontSize='""" + wch_font_size + """';} } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)


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


###### Config
st.markdown(
    """<style>
div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 32px;
}
    </style>
    """, unsafe_allow_html=True
)


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
            
        st.divider()    
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
        "Explore customer's data  ",
        "Main factors in the model decision  ",
        "Compare customer to others",
    ]
)
# Change tab font-size
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.15rem;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

with tab1:
    if customer_id == 'XXXXXX':
        st.write('')
    else:
        col1, _, col2 = st.columns([1, 0.2, 1])
        with col2:
            st.write(
                "If no features are selected, you can scroll down the table "
                "to explore the whole data we own about the customer.\n\n"
                "Otherwise, you are invited to select multiple features with "
                "the selector right below.\n\n"
                "*Note* : In the selector, Features are order by their "
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
        
with tab2:        
    # st.write("## Major factors in the decision")
    
    st.image("https://i.stack.imgur.com/Ftxu7.png")
    
    st.write(
        "A value pushing the decision to the left helps in getting"
        " a loan. Otherwise, it plays against the customer."
    )    

with tab3:
    # 2 columns to select the feature and the group to compare to.
    col1, col2 = st.columns([1, 1])
    with col2:
        st.write("")
        st.write("")
        ft = st.selectbox(
            label="Which feature to compare?",
            options=fts_by_model_importance,
        )
        ChangeWidgetFontSize('Which feature to compare?', '18px')
        
        
    with col1:
        sel_group = st.radio(
            "Who to compare too?",
            [
                "Everyone.",
                "People with same education, occupation and age range.",
                "People with same credit duration, income type and education."
            ],
        )
        ChangeWidgetFontSize('Who to compare too?', '18px')
    
    # Define features to groupby
    group1 = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'AGE_RANGE']
    group2 = ['CREDIT_TO_ANNUITY_GROUP', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']   
    
    # Select the others in function of the chosen group.
    match sel_group:
        case "everyone":
            st.write("everyone sel")
        case "people with same education, occupation and age range":
            st.write("group1 sel")
        case "people with same credit duration, income type and education":
            st.write("group2 sel")
                            

