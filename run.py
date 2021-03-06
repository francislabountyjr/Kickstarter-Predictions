import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import joblib

# Imports from this application
from app import app, server
from pages import index, predictions, insights, process

# Navbar docs: https://dash-bootstrap-components.opensource.faculty.ai/l/components/navbar
navbar = dbc.NavbarSimple(
    brand='Kickstarter',
    brand_href='/',
    # children=[
    #     # dbc.NavItem(dcc.Link('Predictions', href='/predictions', className='nav-link')),
    #     dbc.NavItem(
    #         dcc.Link('Insights', href='/insights', className='nav-link')),
    #     dbc.NavItem(dcc.Link('Process', href='/process', className='nav-link')),
    # ],
    sticky='top',
    color='black',
    light=False,
    dark=True
)


# Layout docs:
# html.Div: https://dash.plot.ly/getting-started
# dcc.Location: https://dash.plot.ly/dash-core-components/location
# dbc.Container: https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # navbar,
    dbc.Container(id='page-content', className='mt-4'),
    html.Hr(),
    # footer
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return index.layout
    # elif pathname == '/predictions':
    #     return predictions.layout
    # elif pathname == '/insights':
    #     return insights.layout
    # elif pathname == '/process':
    #     return process.layout
    else:
        return dcc.Markdown('## Page not found')


# Run app server: https://dash.plot.ly/getting-started
if __name__ == '__main__':
    app.run_server(debug=True)
