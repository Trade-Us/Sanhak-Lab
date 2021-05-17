import dash
import dash_core_components as dcc
import dash_html_components as html


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def textResultDiv():
    cluster_num = "7"
    data_num = "150"
    time = "30"
    used_algo = "DTW"   
    textdata = html.Div(children=[
        html.H3(children='군집 개수 : '+cluster_num),
        html.H3(children='군집별 데이터 개수 : '+data_num),
        html.H3(children='총 소요 시간 : '+time),
        html.H3(children='사용된 알고리즘 : '+used_algo),
    ])
    return textdata