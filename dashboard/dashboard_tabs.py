import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time
from group import Group

DATA_URI = (
    '/home/louberehc/OCR/projets/7_scoring_model/'
    'pickle_files/features_sample.pkl'
)

@st.cache_data
def get_data():
    """
    Get the data about of all customers and cache it.
    Convert values to float64 to avoid later problems.
    """
    df = pd.read_pickle(DATA_URI).astype("float64")
    target = df.pop('TARGET')
    return df, target

@st.cache_data
def instantiate_groups():
    group_list = [
        Group('group 1', 'Everyone.', None),
        Group(
            name='group 2',
            description="People with same education, occupation and age range.",
            grouper=['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'AGE_RANGE']
        ),
        Group(
            name='group 3',
            description="People with same credit duration, income type and education.",
            grouper=['CREDIT_TO_ANNUITY_GROUP', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']
        ),
    ]
    return group_list
        

@st.cache_data
def get_customer_score(customer_data):
    time.sleep(3)
    return 0.27, 0.16    


def ChangeWidgetFontSize(wgt_txt, wch_font_size = '12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.fontSize='""" + wch_font_size + """';} } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)
    
    


###### Config
# Change tab font-size
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.15rem;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

# To draw the score bar
svg = """
<svg width="350" height="100" viewBox="0 0 120 30" version="1.1" xmlns="http://www.w3.org/2000/svg">
<defs>
    <linearGradient id="Gradient1">
    <stop class="stop1" offset="0%" />
    <stop class="stop2" offset="40%" />
    <stop class="stop3" offset="50%" />
    <stop class="stop4" offset="60%" />
    <stop class="stop5" offset="100%" />
    </linearGradient>
</defs>
<style>
    #rect1 {
    fill: url(#Gradient1);
    }
    .stop1 {
    stop-color: mediumblue;
    }
    .stop2 {
    stop-color: royalblue;
    }
    .stop3 {
    stop-color: lightgray;
    }
    .stop4 {
    stop-color: indianred;
    }
    .stop5 {
    stop-color: firebrick;
    }
</style>

<rect id="rect1" x="0" y="15" rx="2" ry="2" width="100" height="15" />
<path d="M50,15 v+15" fill="yellow" stroke="black" stroke-width="1" />
<text x="4" y="21" font-size="5px">No repayment</text>
<text x="6" y="27" font-size="5px">difficulties</text>
<text x="70" y="21" font-size="5px">Repayment</text>
<text x="72" y="27" font-size="5px">difficulties</text>
<polygon points="40,13 38,9 42,9" style="fill:orange;stroke:black;stroke-width:0.2" />
<text x="23" y="5" font-size="5px" fill="orange">Customer score</text>
</svg>
"""


###### Global variable definitions
df, target = get_data()
valid_customer_ids = list(df.index)
fts_by_model_importance = df.columns # TODO: change with model global interp.

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
        st.write("### Model prediction")
        # Print the model decision
        if customer_score < model_threshold:
            st.write("Credit application ACCEPTED.")
        else:
            st.write("Credit application REJECTED.")
            
        st.divider()    
        # Print gauge to tell if close from the decision boundary
        # place for the gauge
        st.write("### Score relative to the decision boundary")
        
        st.markdown(svg, unsafe_allow_html=True)
        
            
        
# start tabs       
tab1, tab2, tab3 = st.tabs(
    [
        "Explore customer's data",
        "Main factors in the model decision",
        "Compare customer to others",
    ]
)

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
    if customer_id == 'XXXXXX':
        st.write('')
    else:        
        # st.write("## Major factors in the decision")
        
        st.image("https://i.stack.imgur.com/Ftxu7.png")
        
        st.write(
            "A value pushing the decision to the left helps in getting"
            " a loan. Otherwise, it plays against the customer."
        )    
        
########################################################################
# Select a feature and a group and plot the distribution of the feature
# inside the group + pinpoint the customer value.
########################################################################
with tab3:
    if customer_id == 'XXXXXX':
        st.write('')
    else:
        # Group definitions once and for all thanks to a function with 
        # a cache_data decorator (avoid reload and duplication)
        groups = instantiate_groups()
        
        # Widgets to select the feature and the group to compare 
        # to inside 2 columns.
        col1, col2 = st.columns([1, 1])
        with col2:
            st.write("")
            st.write("")
            sel_ft = st.selectbox(
                label="Which feature to compare?",
                options=fts_by_model_importance,
            )
            ChangeWidgetFontSize('Which feature to compare?', '18px')
            
        with col1:
            sel_group_name = st.radio(
                label="Who to compare too?",
                options=[g.name for g in groups],
                captions=[g.description for g in groups]
            )
            ChangeWidgetFontSize('Who to compare too?', '18px')
        
        # Get the Group class and use it to plot.
        sel_group = [g for g in groups if g.name == sel_group_name][0]
        fig = sel_group.plot_feature_kde_with_client_value(
            features_df=df,
            target=target,
            customer_id=customer_id,
            feature_name=sel_ft,
        )
        st.pyplot(fig)