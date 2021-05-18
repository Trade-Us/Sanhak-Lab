import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq


def kmeans_layout():
    return html.Div(id='kmeans-param', children=[
        dcc.Store(id='store-kmeans-param', data=[]),
        html.H5("KMeans Parameters"),
        html.Label('Cluster 개수'),
        dcc.Input(id='number-of-cluster', min=2, value=2, type='number'),
        html.Label('Tolerance, default = 1e-4'),
        dcc.RadioItems(id='tolerance', 
            options=[
                {'label': '1e-4', 'value': 1e-3},
                {'label': '1e-4', 'value': 1e-4},
                {'label': '1e-4', 'value': 1e-5},
            ], value=1e-4),
        html.Label('KMeans를 시도해볼 횟수'),
        dcc.Input(id='try-n-init', min=1, value=10, type='number'),
        html.Label('Kmeans가 알고리즘 안에 반복되는 최대 횟수'),
        dcc.Input(id='try-n-kmeans', min=10, value=300, type='number'),
        html.Label('중심 랜덤으로 지정하기'),
        daq.BooleanSwitch(id='random-center', on=False, label="랜덤 사용", labelPosition='top'),
        html.Hr()
    ])
def dbscan_layout():
    return html.Div(id='dbscan-param', children=[
        dcc.Store(id='store-dbscan-param', data=[]),
        html.H5("DBSCAN Parameters"),
        html.Label('Epsilon 크기'),
        dcc.Input(id='dbscan-epsilon', min=0, max=1, value=0.5, type='number'),
        html.Label('min-sample 크기(정수)'),
        dcc.Input(id='dbscan-min-sample', min=1, value=5, type='number'),
        html.Hr()
    ])
def hierarchy_layout():
    return html.Div(id='hierarchy-param', children=[
        dcc.Store(id='store-hierarchy-param', data=[]),
        html.H5("Hierarchy Parameters"),
        html.Label('Cluster 개수'),
        dcc.Input(id='number-of-cluster', min=2, value=2, type='number'),
        html.Label('n-init'),
        dcc.Input(id='try-n-init', min=1, value=10, type='number'),
        html.Label('linkage'),
        dcc.Dropdown(
            id='linkage',
            options=[
                {'label': 'ward', 'value': 'ward'}
            ],
            value='ward'),
        html.Hr()
    ])
def time_sereies_kmeans_layout():
    layout = html.Div([
            html.Div(id='hidden-tsk-div', style={'display':'none'}),
            dcc.Store(id='store-distance-algorithm', data=[]),
            html.Label('Cluster 개수'),
            dcc.Input(id='number-of-cluster', min=2, value=2, type='number'),
            html.Label('거리계산 알고리즘'),
            dcc.Dropdown(
                id='distance-algorithm', 
                options=[
                    {'label':'Eucleadean', 'value':'EUC'},
                    {'label':'DTW', 'value':'DTW'},
                    {'label':'Soft-DTW', 'value':'SDT'}
                ], value='DTW'),
            html.H5('dtw, soft-dtw 경우 설정하는 파라미터: barycenter, gamma'),
            html.Label('path 구하는 알고리즘 돌리는 횟수'),
            dcc.Input(id='try_n_barycenter', value=100, min=100, max=200, type='number'),
            html.Label('Metric Gammas 높을 수록 부드러우지지만, 시간이 걸림'),
            dcc.Slider(id='metric_gamma', min=0, max=1, step=0.1,
            marks={i/10: '{}'.format(i/10) if i != 0 else '0' for i in range(0, 11)},
            value=0.1),
            html.Label('Try N Times for another center'),
            dcc.Input(id='try-n-init', min=1, value=10, type='number'),
        ], style={'columnCount': 1})
    return layout