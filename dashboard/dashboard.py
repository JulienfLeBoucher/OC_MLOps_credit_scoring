import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
# import json
from urllib.parse import urljoin
import time
from group import Group
import random
import base64
import visual_elem

API_ROOT = "http://localhost:8435/"




ICON_PATH = "./images/tab_icon.png"
APP_NAME = "Credit attribution explorer"

# Set icon and name of the tab in browser.
st.set_page_config(
    page_title=APP_NAME,
    page_icon=ICON_PATH,
    layout="centered",
    initial_sidebar_state="auto",
)

DATA_URI = (
    '/home/louberehc/OCR/projets/7_scoring_model/'
    'pickle_files/reduced_data.pkl'
)

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)
    
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
def get_customer_score_fake(customer_data):
    time.sleep(3)
    return random.randint(0, 100), 50


@st.cache_data
def get_customer_score(customer_id):
    request_url = urljoin(API_ROOT, f'prediction/{customer_id}')
    r = requests.get(request_url)
    return r
    


def ChangeWidgetFontSize(wgt_txt, wch_font_size = '12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.fontSize='""" + wch_font_size + """';} } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)
    
    






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
        
        with st.spinner('Wait for the model decision...'):
            # Send request to the model API 
            # customer_score, model_threshold = get_customer_score_fake(customer_data)
            r = get_customer_score(customer_id)
            st.json(r.json())
        # Print the model decision
        st.divider()
        st.write("### Model decision")
        
        # if customer_score < model_threshold:
        #     render_svg(visual_elem.ACCEPTED_SVG)
        # else:
        #     render_svg(visual_elem.REJECTED_SVG)
        # # Print gauge to tell if close from the decision boundary
        # st.divider()
        # st.write("### Model perception")
        # st.markdown(
        #     visual_elem.draw_score_bar(customer_score),
        #     unsafe_allow_html=True
        # )
        
        
# Application title
render_svg(visual_elem.TITLE_SVG)
        
# start tabs       
tab1, tab2, tab3 = st.tabs(
    [
        "Explore customer's data",
        "Main factors in the model decision",
        "Compare customer to others",
    ]
)
# Change tab font-size
st.markdown(visual_elem.tab_font_size, unsafe_allow_html=True)


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
        # Group selection
        sel_group_name = st.radio(
                    label="Who to compare too?",
                    options=[g.name for g in groups],
                    captions=[g.description for g in groups]
                )
        ChangeWidgetFontSize('Who to compare too?', '18px')
            
        with st.expander('Univariate analysis'):
            # Feature selection
            sel_ft = st.selectbox(
                label="Which feature to compare?",
                options=fts_by_model_importance,
            )
            ChangeWidgetFontSize('Which feature to compare?', '18px')
            # Get the Group class and use it to plot.
            sel_group = [g for g in groups if g.name == sel_group_name][0]
            fig = sel_group.plot_feature_kde_with_client_value(
                features_df=df,
                target=target,
                customer_id=customer_id,
                feature_name=sel_ft,
            )
            st.pyplot(fig)
            
        with st.expander('Bi-variate analysis'):
            ChangeWidgetFontSize('Bi-variate analysis', '18px')
            col1, _, col2 = st.columns([1, 0.2, 1])
            with col1:
                ft_x = st.selectbox(
                    label="Abscissa feature",
                    options=fts_by_model_importance,
                )
                ChangeWidgetFontSize('Abscissa feature', '18px')
                
            with col2:
                ft_y = st.selectbox(
                    label="Ordinate feature",
                    options=fts_by_model_importance,
                )
                ChangeWidgetFontSize('Ordinate feature', '18px')

            