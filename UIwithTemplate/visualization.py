import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc

data = [
    [ 
        [5,8,2,2],
        [5,2,4,6,5,4,3,7,7,8,7,4],
        [4,11,2,1,2,7],
        [7,4,5,7,4,0,7,2,3,1]
    ],
    [
        [4,5,3,4,4,4,2,0,6,2],
        [4,4,0,9],
        [2,4,3,5,3,3,6,7,3,8],

    ],
    [
        [4,2,3,2,4,3,4],
        [7,6,5,6,6,5,4],
        [3,2,3,4,5,4],
        [4,4,0,9],
        [2,4,3,5,3,3,6,7,3,8],
        [5,4,5,5]
    ],
    [
        [1,2,6,2,3,3,1],
        [2,6,4,6,6,3,4],
        [4,2,8,3,5,7],
        [5,4,5,5],
        [4,3,4,8,9,2,2,2]
    ]
]

#시계열셋 최소 길이 리스트형태로 담음
min_lens=[]
data_len=len(data)
for i in range(0,data_len):
    min=len(data[i][0])
    for j in range(0,len(data[i])):
        if len(data[i][j]) < min:
            min = len(data[i][j])
    min_lens.append(min)

#시계열 셋 길이 통일
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
result_re=[]
for i in range(0,data_len):
    result_ = TimeSeriesResampler(sz=min_lens[i]).fit_transform(data[i])
    result_=result_.reshape(len(data[i]),min_lens[i])
    result_re.append(result_)

#수치형 변수 정규화
from sklearn.preprocessing import StandardScaler
result_norm=[]
for i in range(0, data_len):
    norm = StandardScaler().fit_transform(result_re[i])
    result_norm.append(norm)


#주성분 분석 실시하기
from sklearn.decomposition import PCA

#PCA 객체 생성 (주성분 갯수 2개 생성)
pca = PCA(n_components=2)
result_pca=[]
for i in range(0,data_len):
    pca_ = pca.fit_transform(result_norm[i])
    result_pca.append(pca_)

#그래프 그리기
fig = go.Figure()

for i in range(0,data_len):
    fig.add_trace(go.Scatter(
        x=result_pca[i][:,0], y= result_pca[i][:,1],
        mode='markers', name='cluster'+str(i)
        )
    )

def pca_show():
    graph = html.Div(style={}, children=[
        html.Div(["2-DIM VISUALIZATION"], className='subtitle'),
        html.Div(
            [html.Div(
                dcc.Graph(id='pca_show', figure=fig))
            ]
        )
    ])

    return graph