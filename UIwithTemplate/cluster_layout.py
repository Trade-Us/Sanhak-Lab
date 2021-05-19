import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq


def kmeans_layout():
    return html.Div(id='kmeans-param', children=[
        dcc.Store(id='store-kmeans-param', data=[]),
        html.H4("KMeans Parameters"),
        html.H6('Cluster 개수'),
        dcc.Input(id='number-of-cluster', min=2, value=2, type='number'),
        html.H6('Tolerance, default = 1e-4'),
        dcc.RadioItems(id='tolerance', 
            options=[
                {'label': '1e-3', 'value': 1e-3},
                {'label': '1e-4', 'value': 1e-4},
                {'label': '1e-5', 'value': 1e-5},
            ], value=1e-4),
        html.H6('KMeans를 시도해볼 횟수'),
        dcc.Input(id='try-n-init', min=1, value=10, type='number'),
        html.H6('Kmeans가 알고리즘 안에 반복되는 최대 횟수'),
        dcc.Input(id='try-n-kmeans', min=10, value=300, type='number'),
        html.H6('중심 랜덤으로 지정하기'),
        daq.BooleanSwitch(id='random-center', on=False, label="랜덤 사용", labelPosition='top'),
        html.Hr()
    ])
def dbscan_layout():
    return html.Div(id='dbscan-param', children=[
        dcc.Store(id='store-dbscan-param', data=[]),
        html.H4("DBSCAN Parameters"),
        html.Div([
            html.H6('Epsilon 크기'),
            dcc.Input(id='dbscan-epsilon', min=0, max=1, value=0.5, type='number'),
        ],className='twodiv'),
        html.Div([
            html.H6('min-sample 크기(정수)'),
            dcc.Input(id='dbscan-min-sample', min=1, value=5, type='number'),
        ],className='twodiv'),
        html.Hr()
    ])
def hierarchy_layout():
    return html.Div(id='hierarchy-param', children=[
        dcc.Store(id='store-hierarchy-param', data=[]),
        html.H4("Hierarchy Parameters"),
        html.H6('Cluster 개수'),
        dcc.Input(id='number-of-cluster', min=2, value=2, type='number'),
        html.Div([
            html.H6('n-init'),
            dcc.Input(id='try-n-init', min=1, value=10, type='number'),
        ], className='twodiv'),
        html.Div([
            html.H6('linkage'),
            dcc.Dropdown(
                id='linkage',
                options=[
                    {'label': 'ward', 'value': 'ward'}
                ],value='ward'),
        ], className='twodiv'),
        html.Hr()
    ])
def time_sereies_kmeans_layout():
    layout = html.Div([
            html.Div(id='hidden-tsk-div', style={'display':'none'}),
            dcc.Store(id='store-distance-algorithm', data=[]),
            html.H4("TimeseriesKmeans Parameters"),
            html.H6('Cluster 개수'),
            dcc.Input(id='number-of-cluster', min=2, value=2, type='number'),
            html.H6('거리계산 알고리즘'),
            dcc.Dropdown(
                id='distance-algorithm', 
                options=[
                    {'label':'Eucleadean', 'value':'EUC'},
                    {'label':'DTW', 'value':'DTW'},
                    {'label':'Soft-DTW', 'value':'SDT'}
                ], value='DTW'),
            html.H6('path 구하는 알고리즘 돌리는 횟수'),
            dcc.Input(id='try_n_barycenter', value=100, min=100, max=200, type='number'),
            html.H6('Metric Gammas'),
            html.Label('높을 수록 부드러우지지만, 시간이 걸림'),
            dcc.Slider(id='metric_gamma', min=0, max=1, step=0.1,
            marks={i/10: '{}'.format(i/10) if i != 0 else '0' for i in range(0, 11)},
            value=0.1),
            html.H6('Try N Times for another center'),
            dcc.Input(id='try-n-init', min=1, value=10, type='number'),
            html.Hr()
        ], style={'columnCount': 1})
    return layout