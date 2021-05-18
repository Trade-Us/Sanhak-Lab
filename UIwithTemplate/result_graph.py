import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go




# graph: dictionary 형태의 데이터
# n: 한 군집 내 시계열 데이터 개수
# label: 딕셔너리 keys(list형태)
# color: 그래프 line 색

def makeGraph_dictionary(graph, n, label, color):
    df=[]
    for i in range(0, n):
        df.append(graph[label[i]])

    fig = go.Figure()
    for i in range(0, n):
        fig.add_trace(go.Scatter(y=df[i], name=label[i], line=dict(color=color)))

    return fig

def makeGraph_Cluster(graph, color):                    
    
    fig = go.Figure()
    for i in range(0, len(graph)):
        fig.add_trace(go.Scatter(y=graph[i],  line=dict(color=color), showlegend=False))
    return fig
    
# figure: makeGraph()를 이용해 만든 그래프
# label: 그래프 이름
def updateLayout(figure, name, yaxis='value'):
    figure.update_layout(
        title=name,
        yaxis_title=yaxis,
    )

def makeGraph_Detail(graph, color):                    
    
    fig = go.Figure(data=go.Scatter(y=graph,  line=dict(color=color)))
    return fig
    
# figure: makeGraph()를 이용해 만든 그래프
# label: 그래프 이름
def updateLayout_Detail(figure, name, yaxis='value'):
    figure.update_layout(
        title=name,
        yaxis_title=yaxis
    )

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

colors = {
    'background': 'dimgray',
    'text': 'white'
}



def graphCluster(GG):
    figs=[]
    for i in range(0,len(GG)):
        figs.append(makeGraph_Cluster(GG[i], 'teal'))
        updateLayout(figs[i], 'cluster'+str(i))
    graph = html.Div(style={ }, children=[
        html.Div(["CLUSTERS"], className='subtitle'),
        html.Div(
            [html.Div(
                dcc.Graph(id=f'GC{i}', figure=fig), 
                className='graph graph-hover'
                ) for i, fig in enumerate(figs)
            ], className='graphdiv'
        )
    ], className='clusters')

    return graph

def graphDetail(nth_cluster, num_graph, GG):
    figs=[]
    for i in range(0, num_graph):
        figs.append(makeGraph_Detail(GG[nth_cluster][i], 'firebrick'))
        updateLayout(figs[i], 'Cluster0_randomData'+str(i))

    graph = html.Div(style={'height': "500px"}, children=[
        html.Div(
            [html.Div(
                dcc.Graph(id=f'GD{i}', figure=fig), 
                className='graph'
                ) for i, fig in enumerate(figs)
            ]
        )
    ])
    
    return graph

def graphBig(nth_cluster, num_graph, GG):
    fig = []
    fig.append(makeGraph_Cluster(GG[nth_cluster][:num_graph], 'teal'))
    updateLayout(fig[0], 'cluster'+str(nth_cluster))

    graph = html.Div(style={}, children=[
        html.Div(
            [html.Div(
                [dcc.Graph(
                    id='GB1',
                    figure=fig[0]
                )]),
            ]
        )
    ])

    return graph

def textResultDiv(num_cluster, num_tsdatas_per_cluster, siluet_score, used_algorithm):
    textdata = html.Div(children=[
        html.Div(["SUMMARIZATION"], className='subtitle'),
        html.Div([
            html.Div(children=f'군집 개수 : {num_cluster}개'),
            html.Hr(),
            html.Div(children=f'군집별 데이터 개수 : {num_tsdatas_per_cluster}'),
            html.Hr(),
            html.Div(children=f'실루엣 점수 : {siluet_score}'),
            html.Hr(),
            html.Div(children=f'사용된 알고리즘 : {used_algorithm}'),
            html.Hr(),
        ], className='textbox')
    ])
    return textdata
