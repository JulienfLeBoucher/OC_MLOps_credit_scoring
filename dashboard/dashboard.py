import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import io
import json
from urllib.parse import urljoin
import time
from group import Group
import random
import base64
import visual_elem

DEBUG = True

#
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

# DATA_URI = (
#     '/home/louberehc/OCR/projets/7_scoring_model/'
#     'pickle_files/reduced_data.pkl'
# )

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)
    
@st.cache_data
def get_data():
    """
    Request data features about all customers to the API to use for 
    comparison, return it as a dataframe and cache it.
    
    There, all customers are used because I am only using a very light
    sample of data to avoid cost while deploying the API.
    It could be interesting to sample in the function if the data
    becomes larger.
    """
    request_url = urljoin(API_ROOT, 'data')
    r = requests.get(request_url)
    # Retrieve the json string because I found no way to convert the r.json()
    # to a dataframe directly.
    t = r.text
    return pd.read_json(io.StringIO(t))

@st.cache_data
def get_target():
    """
    Request data features about all customers to the API to use for 
    comparison, return it as a DataFrame and cache it.
    
    There, all customers are used because I am only using a very light
    sample of data to avoid cost while deploying the API.
    It could be interesting to sample in the function if the data
    becomes larger.
    """
    request_url = urljoin(API_ROOT, 'target')
    r = requests.get(request_url)
    # Retrieve the json string because I found no way to convert the r.json()
    # to a dataframe directly.
    t = r.text
    return pd.read_json(io.StringIO(t))


@st.cache_data
def get_customer_info(customer_id):
    """ Request a model inference via the API and return the response."""
    request_url = urljoin(API_ROOT, f'prediction/{customer_id}')
    r = requests.get(request_url)
    return r


@st.cache_data
def get_model_info():
    """ Request model information and return the response."""
    request_url = urljoin(API_ROOT, 'model_info')
    r = requests.get(request_url)
    return r


def convert_requests_response_to_dict(response):
    return json.loads(response.text)


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


def ChangeWidgetFontSize(wgt_txt, wch_font_size = '12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.fontSize='""" + wch_font_size + """';} } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)
    

def map_model_output_to_bar_position(score, decision_threshold):
    """ Map arbitrarily the model proba prediction in [0, 1] to its 
    position on a score bar which goes from 0 to 100 having a decision
    boundary representation in the middle when the model decision threshold
    is not necessarily at 0.5.
    
    It could be done the smart way observing scores of true and false positive
    in order to identify where the model has a nice certainty (extremes)
    and where it is closer to the boundary decision.
    """
    # Arbitrary chosen values
    left_uncertain = model_threshold - 0.03 
    right_uncertain = model_threshold + 0.03 
    delta_uncertain = right_uncertain - left_uncertain
    if score < left_uncertain:
        return (score / left_uncertain * 40)
    elif (score > left_uncertain) and (score < right_uncertain):
        return (40 + ((score-left_uncertain) / delta_uncertain)*20)
    else:
        return (60 + (score-right_uncertain)/(1-right_uncertain)*40)


###### Global requests
features = get_data()
target = get_target()
model_info = convert_requests_response_to_dict(get_model_info())

# Extract some useful values
model_threshold = model_info['decision_threshold']
valid_customer_ids = list(features.index)

fts_by_model_importance = features.columns # TODO: change with model global interp.

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
            features
            .loc[customer_id, :]
            .T
        )
        customer_data.index.name = 'Feature name'
        customer_data.name = 'Value'
        
        # Send request to the model API to get customer prediction
        # as a json object
        customer_info = convert_requests_response_to_dict(
            get_customer_info(customer_id)
        )
            
        # Print the model decision
        st.divider()
        st.write("### Model decision")
        if customer_info['class'] == 'no_risk':
            render_svg(visual_elem.ACCEPTED_SVG)
        else:
            render_svg(visual_elem.REJECTED_SVG)
            
        # Print gauge to tell if close from the decision boundary
        st.divider()
        st.write("### Model perception")
        score_on_bar = map_model_output_to_bar_position(
            score=customer_info['proba_risk_class'],
            decision_threshold=model_threshold
        )
        st.markdown(
            visual_elem.draw_score_bar(score_on_bar),
            unsafe_allow_html=True
        )
        
        if DEBUG:
            st.divider()
            with st.expander('Technical information'):
                st.write("Model information:")
                st.write(model_info)
                st.write("Customer information:")
                st.write(customer_info)

### Dashboard central part :
# Application title
render_svg(visual_elem.TITLE_SVG)
        
# Tabs creation       
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
        # a cache_data decorator
        groups = instantiate_groups()
        # Group selection
        sel_group_name = st.radio(
                    label="Who to compare too?",
                    options=[g.name for g in groups],
                    captions=[g.description for g in groups]
                )
        ChangeWidgetFontSize('Who to compare too?', '18px')
        
        ChangeWidgetFontSize('Univariate analysis', '20px')
        with st.expander('Univariate analysis'):
            # Feature selection
            sel_ft = st.selectbox(
                label="Which feature to compare?",
                options=fts_by_model_importance,
            )
            ChangeWidgetFontSize('Which feature to compare?', '18px')
            # Get the select group and use it to plot.
            sel_group = [g for g in groups if g.name == sel_group_name][0]
            fig = sel_group.plot_feature_kde_with_client_value(
                features_df=features,
                target=target,
                customer_id=customer_id,
                feature_name=sel_ft,
            )
            if fig is not None:
                st.pyplot(fig)
            else: 
                st.write(
                    "There are to few people in the group to "
                    "plot a meaningful visualization.")
            
            st.markdown(visual_elem.plot_note)
            
        ChangeWidgetFontSize('Bi-variate analysis', '20px')
        with st.expander('Bi-variate analysis'):
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

            