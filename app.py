# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


## data cleaning
df = pd.read_excel("./sed17-sr-tab007 transpose.xlsx", index_col=[0,1,2,3])
state_names = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District ", "of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
us_state_abbrev = {
'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY', 'Puerto Rico': 'PR'}
df_new = df.filter(items=state_names).reset_index()

data = df_new.loc[:, df_new.columns != 'All institutions']
data = data.iloc[1:, :]
data.rename(columns={'State or location and institution': 'Field'}, inplace=True)
data = data.T



def data_cleaning(df, i):
    df = data.rename(columns=data.iloc[i])
    df.drop(df.index[0:3], inplace=True)
    df = df.groupby(by=df.columns, axis=1).sum().reset_index()
    df.rename(columns={'index':'State'}, inplace=True)
    df = pd.melt(df, id_vars=['State'])
    df.rename(columns={'variable':'Field'}, inplace=True)
    df = df.sort_values(by=['Field','value'], ascending=False)
    df['State_ABBR'] = df['State'].map(us_state_abbrev).fillna(df['State'])
    return df


df0 = data_cleaning(data, 0)
fig0 = px.bar(df0, x="State", y="value", color="Field", barmode="group")

df1 = data_cleaning(data, 1)
fig1 = px.bar(df1, x="State", y="value", color="Field", barmode="group")

fig_map = px.choropleth(df0, locationmode = 'USA-states', locations='State_ABBR', color='value',
                           color_continuous_scale="Viridis",
                           scope="usa",
                           labels={'value':'number of phDs'})
                           

## HTML
app.layout = html.Div(children = [
    html.Div([
        html.H4(children='Choose Your Field'),
            
        dcc.Dropdown(
            id='dropdown', 
            options=[
                {'label': 'Science', 'value': 'Science'},
                {'label': 'Engineering', 'value': 'Engineering'}
            ], 
            value='Science',
        ),
        html.Div(id='output-container')
    ]),
    html.Div([
        html.H1(children='Number of Science and Engineering Doctorate Degree Granted, by State in 2017',
                style = {'font-family': 'Helvetica',
                          'font-size': '25px',
                          'textAlign': 'center'}
                        ),
        html.Div(children='''
            Hover over to see the statistics!''',
            style = {'font-family': 'Helvetica',
                          'font-size': '15px',
                          'textAlign': 'center'}
                        ),
        dcc.Graph(
            id='graph',
            figure={}
        ),  
    ]),
    html.Div([
        html.H4(children='Choose Your Sub Field'),
            
        dcc.Dropdown(
            id='dropdown2', 
            options=[
                {'label': 'Psychology and social sciences ', 'value': 'Psychology and social sciences '},
                {'label': 'Physical sciences and earth sciences', 'value': 'Physical sciences and earth sciences'},
                {'label': 'Mathematics and computer sciences', 'value': 'Mathematics and computer sciences'},
                {'label': 'Life sciences', 'value': 'Life sciences'}
            ], 
            value='Psychology and social sciences ',
        ),
        html.Div(id='output-container2')
    ]),
    html.Div([
        html.H1(children='Number of Doctorate Degree Granted by Sub Field, by State in 2017',
                style = {'font-family': 'Helvetica',
                          'font-size': '25px',
                          'textAlign': 'center'}
                        ),
        html.Div(children='''
            Hover over to see the statistics!''',            
            style = {'font-family': 'Helvetica',
                          'font-size': '15px',
                          'textAlign': 'center'}
                        ),
        dcc.Graph(
            id='graph1',
            figure=fig0
        ),  
    ]),
])

df1


@app.callback(
    Output('graph','figure'),
    Input('dropdown', 'value')
)
def update_output(value):
    #print(df1)
    df = df0[df0['Field'] == value]
    # print(value)
    fig = px.choropleth(df, locationmode = 'USA-states', locations='State_ABBR', color='value',
                           color_continuous_scale="Viridis",
                           scope="usa",
                           labels={'value':'number of phDs'})
    return fig
#'You have selected "{}"'.format(value)

@app.callback(
    Output('graph1','figure'),
    Input('dropdown2', 'value')
)
def update_output2(value):
    df = df1[df1['Field'] == value]
    # print(value)
    fig = px.bar(df, x="State", y="value", color="Field", barmode="group")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
