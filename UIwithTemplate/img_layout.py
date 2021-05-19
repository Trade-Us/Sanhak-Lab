import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq

def rp_layout():
    return html.Div(id='rp-param', children=[
        dcc.Store(id='store-rp-param', data=[]),
        html.H4("RP Parameters"),
        html.H6("Image Size"),
        html.Label("생성할 이미지 크기 결정"),
        dcc.RadioItems(id='image-size', 
            options=[
                {'label': '28', 'value': 28},
                {'label': '96', 'value': 96},
            ], value=28),
        html.H6("Dimension"),
        html.Label("RP 궤적의 차원수를 결정한다. 공간 궤적 좌표 생성에 쓰이는 데이터 개수이다."),
        dcc.Input(id='dimension', value=1, min=1, type='number'),

        html.H6("Time-Delay"),
        html.Label("공간 궤적 좌표 생성시 사용되는 기존 좌표 데이터의 시간 차이를 뜻한다. 따라서 1dim 데이터 사용시 큰 의미가 없다."),
        dcc.Input(id='time-delay', value=1, min=1, type='number'),

        html.H6("Threshold"),
        html.Label("궤적의 거리 최솟값을 설정한다."),
        dcc.Dropdown(id='threshold',
        options=[
            {'label': 'float', 'value':'float'},
            {'label': 'point', 'value':'point'},
            {'label': 'distance', 'value':'distance'},
            {'label': 'None', 'value' : 'None'}
        ], value='F'),
        html.Label("percentage if point or distance"),
        dcc.Slider(id='percentage', min=10, max=60, marks={i: '{}'.format(i) for i in range(10, 61, 10)}, value=1, step=1),
        html.Hr()
    ])

def gaf_layout():
    return html.Div(id='gaf-param', children=[
        dcc.Store(id='store-gaf-param', data=[]),
        html.H4("GAF Parameters"),
        html.H6("Image Size"),
        html.Label("생성할 이미지 크기 결정"),
        dcc.RadioItems(id='image-size', 
            options=[
                {'label': '28', 'value': 28},
                {'label': '96', 'value': 96},
            ], value=28),
        html.H6("method"),
        html.Label("GAF Summation vs GAF Difference"),
        dcc.Dropdown(id='gaf-method',
            options=[
                {'label': 'summation', 'value':'SUM'},
                {'label': 'difference', 'value':'DIF'}
            ], value='SUM'),
        html.Hr()
    ])
    
def mtf_layout():
    return html.Div(id='mtf-param', children=[
        dcc.Store(id='store-mtf-param', data=[]),
        html.H4("MTF Parameters"),
        html.H6("Image Size"),
        html.Label("생성할 이미지 크기 결정"),
        dcc.RadioItems(id='image-size', 
            options=[
                {'label': '28', 'value': 28},
                {'label': '96', 'value': 96},
            ], value=28),
        html.H6("Bins"),
        html.Label("number of bins (size of alphabet)"),
        dcc.Input(id='mtf-n-bins', value=5, min=2, max=10, type='number'),
        html.Label("Strategy"),
        dcc.Dropdown(id='mtf-strategy',
            options=[
                {'label': 'quantile', 'value':'QUN'},
                {'label': 'uniform', 'value':'UNI'},
                {'label': 'normal', 'value':'NOR'}
            ], value='QUN'),
        html.Hr()
    ])