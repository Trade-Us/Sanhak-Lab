import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq

def autoencoder_layout():
    return html.Div(id='autoencoder-param', children=[
        dcc.Store(id='store-autoencoder-param', data=[]),
        html.H5("Autoencoder Parameters"),
        html.Label('Batch Size'),
        dcc.RadioItems(id='autoencoder-batch-size', 
            options=[
                {'label': '32', 'value':32},
                {'label': '64', 'value':64}
            ], value=32),
        html.Label('learning rate'),
        dcc.Slider(id='autoencoder-learning-rate', min=1, max=3, marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(1, 4)}),
        html.Label('loss function'),
        dcc.Dropdown(id='autoencoder-loss-function',
            options=[
                {'label': 'rmse', 'value':'RMSE'},
                {'label': 'binary crossentropy', 'value':'BNCP'}
            ], value='RMSE'),
        html.Label('activation function'),
        dcc.Dropdown(id='autoencoder-activation-function',
            options=[
                {'label': 'sigmoid', 'value':'sigmoid'},
            ], value='sigmoid'),
        html.Hr()
    ])
def wavelet_layout():
    return html.Div(id='wavelet-param', children=[
        dcc.Store(id='store-wavelet-param', data=[]),
        html.H5("Wavelet Parameters"),
        html.Label('wavelet function'),
        dcc.Input(id='wavelet-function', value='bd2'),
        html.Label('iteration to half data'),
        dcc.Input(id='iteration-make-half-dim', min=2, value=2, type='number'),
        html.Hr()

    ])
def timeseries_resampler_layout():
    return html.Div(id='ts-resampler-param', children=[
        dcc.Store(id='store-ts-resampler-param', data=[]),
        html.H5("Timeseries Resampler Parameters"),
        html.Label('dimension to reduce yout ts data'),
        dcc.Input(id='ts-resampler-dim', min=2, value=2, type='number'),
        html.Hr()

    ])
    