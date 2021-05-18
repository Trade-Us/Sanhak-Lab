import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq

def autoencoder_layout():
    return html.Div(id='autoencoder-param', children=[
        dcc.Store(id='store-autoencoder-param', data=[]),
        html.H4("Autoencoder Parameters"),
        html.H6('Batch Size'),
        dcc.RadioItems(id='autoencoder-batch-size', 
            options=[
                {'label': '32', 'value':32},
                {'label': '64', 'value':64}
            ], value=32),
        html.H6('Learning Rate'),
        dcc.Slider(id='autoencoder-learning-rate', min=1, max=3, marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(1, 4)}),
        html.H6('Loss Function'),
        dcc.Dropdown(id='autoencoder-loss-function',
            options=[
                {'label': 'rmse', 'value':'RMSE'},
                {'label': 'binary crossentropy', 'value':'BNCP'}
            ], value='RMSE'),
        html.H6('Activation Function'),
        dcc.Dropdown(id='autoencoder-activation-function',
            options=[
                {'label': 'sigmoid', 'value':'sigmoid'},
            ], value='sigmoid'),
        html.Hr()
    ])
def wavelet_layout():
    return html.Div(id='wavelet-param', children=[
        dcc.Store(id='store-wavelet-param', data=[]),
        html.H4("Wavelet Parameters"),
        html.H6('Wavelet Function'),
        dcc.Input(id='wavelet-function', value='bd2'),
        html.H6('Iteration to half data'),
        dcc.Input(id='iteration-make-half-dim', min=2, value=2, type='number'),
        html.Hr()

    ])
def timeseries_resampler_layout():
    return html.Div(id='ts-resampler-param', children=[
        dcc.Store(id='store-ts-resampler-param', data=[]),
        html.H4("Timeseries Resampler Parameters"),
        html.H6('Dimension to reduce your ts data'),
        dcc.Input(id='ts-resampler-dim', min=2, value=2, type='number'),
        html.Hr()

    ])
    