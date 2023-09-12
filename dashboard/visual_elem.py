########################################################################
# SVG
########################################################################
TITLE_SVG = """
<svg viewBox="0 0 600 60" xmlns="http://www.w3.org/2000/svg">
  <style>
  
     /* Note that the color of the text is set with the    *
     * fill property, the color property is for HTML only */
    .title {
      font: italic 45px sans-serif;
      fill: lightgray;
    }
  </style>

  <text x="0" y="50" class="title">Credit attribution explorer</text>
</svg>
"""


REJECTED_SVG = """
<svg viewBox="0 0 240 60" xmlns="http://www.w3.org/2000/svg">
  <style>
    .small {
      font: 16px sans-serif;
      fill: lightgray;
    }
        /* Note that the color of the text is set with the    *
     * fill property, the color property is for HTML only */
    .imp {
      font: 30px sans-serif;
      fill: firebrick;
    }
  </style>

  <text x="0" y="20" class="small">The credit application </text>
  <text x="20" y="50" class="small">is </text>
  <text x="45" y="55" class="imp">REJECTED</text>
</svg>
"""

ACCEPTED_SVG = """
<svg viewBox="0 0 240 60" xmlns="http://www.w3.org/2000/svg">
  <style>
    .small {
      font: 16px sans-serif;
      fill: gray;
    }
        /* Note that the color of the text is set with the    *
     * fill property, the color property is for HTML only */
    .imp {
      font: 30px sans-serif;
      fill: cornflowerblue;
    }
  </style>

  <text x="0" y="20" class="small">The credit application </text>
  <text x="20" y="50" class="small">is </text>
  <text x="45" y="55" class="imp">ACCEPTED</text>
</svg>
"""

def draw_score_bar(score):
    """ Draw a score bar and position the customer score on it."""
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
        stop-color: royalblue;
        }
        .stop2 {
        stop-color: cornflowerblue;
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
    <path d="M50,15 v+15" fill="yellow" stroke="gray" stroke-width="0.5" />
    <text x="4" y="21" font-size="5px">No repayment</text>
    <text x="6" y="27" font-size="5px">difficulties</text>
    <text x="70" y="21" font-size="5px">Repayment</text>
    <text x="72" y="27" font-size="5px">difficulties</text>
    """
    # Manage extreme cases displaying:
    score = score + 3 if score <= 3 else score    
    score = score - 3 if score >= 97 else score
    match score:    
        case score if score <= 15:
            customer_text_abs = 1
        case score if score >= 85:
            customer_text_abs = 68
        case _:
            customer_text_abs = score -15
    
    # customer_text_abs = 83 if score >=83 else score-17
    
    svg += (
        f'<polygon points="{score},13 {score-2},9 {score+2},9"'
        f'style="fill:lightgray;stroke:black;stroke-width:0.2" />'
        f'<text x="{customer_text_abs}" y="7" font-size="5px"' 
        f'fill="lightgray">Customer score</text>'
        f'</svg>'
    )
    return svg

########################################################################
# css
########################################################################
# Change tab font-size
tab_font_size = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.15rem;
    }
</style>
'''