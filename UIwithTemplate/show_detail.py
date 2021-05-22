import dash_html_components as html
import dash_core_components as dcc

# from result_graph import num_clus/ter
def detailGraphOption(num_cluster):
    return [
        html.H4('Options'),
        html.Br(),
        html.H6("군집 선택"),
        dcc.RadioItems(id="nth-cluster",
        options=[
                    {'label': f'Cluster {(i+1)}', 'value': i}
                    for i in range(num_cluster)
                ], value=0),
        html.Hr(),
        html.H6("세부 보기 선택"),
        dcc.Dropdown(id="detail-graph-input", 
        options= [
                    {'label': '개별 그래프 보기', 'value': 'GrDt'},
                    {'label': '확대 보기', 'value': 'GrBg'}
                ], value='GrDt'),
        html.Hr(),
        html.H6("표시되는 그래프 수", id="label-n-graphs"),
        dcc.Input(id='num-of-graphs', min=1, value=1, type='number'),
    ]