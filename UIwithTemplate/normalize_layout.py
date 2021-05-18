import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq

def normalization_layout():
    return html.Div(id='normalization-param', children=[
        dcc.Store(id='store-normalization-param', data=[]),
        html.H4("normalization Parameters"),
        html.Label('normalization method Before Training'),
        dcc.Dropdown(id='normalization-method',
            options=[
                {'label': 'MinMax Scaler', 'value':'MMS'},
                {'label': 'Standard Scaler', 'value':'SSC'},
                {'label': 'Robust Scaler', 'value':'RBS'},
                {'label': 'Max Abs Scaler', 'value':'MAS'},
                {'label': 'Time Series Scaler Mean Variance', 'value':'TSS'}
            ], value='MMS'),
    ])