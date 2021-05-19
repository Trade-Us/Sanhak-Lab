import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc
from tslearn.preprocessing import TimeSeriesResampler
from utils.normalize import Standard
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np



def pca_show(origin_data, labels, num_cluster):
    #시계열 셋 길이 통일
    min=origin_data.dropna(axis='columns')
    min_len=len(min.columns)
    #시계열셋 최소 길이 리스트형태로 담음
    # min_lens=[]
    # data_len=len(data)
    # for i in range(0,data_len):
    #     min=len(data[i][0])
    #     for j in range(0,len(data[i])):
    #         if len(data[i][j]) < min:
    #             min = len(data[i][j])
    #     min_lens.append(min)

    #시계열 셋 길이 통일
    # result_re=[]
    # for i in range(0,data_len):
    #     result_ = TimeSeriesResampler(sz=min_lens[i]).fit_transform(data[i])
    #     result_=result_.reshape(len(data[i]),min_lens[i])
    #     result_re.append(result_)
    result_ = TimeSeriesResampler(sz=min_len).fit_transform(origin_data)
    result_
    result_=result_.reshape(result_.shape[0], min_len)

    result_norm = Standard(pd.DataFrame(result_))
    #수치형 변수 정규화
    # result_norm=[]
    # for i in range(0, data_len):
    #     norm = StandardScaler().fit_transform(result_re[i])
    #     result_norm.append(norm)

    #주성분 분석 실시하기
    pca = PCA(n_components=2) #PCA 객체 생성 (주성분 갯수 2개 생성)
    result_pca = pca.fit_transform(result_norm)
    #주성분 분석 실시하기
    #PCA 객체 생성 (주성분 갯수 2개 생성)
    # pca = PCA(n_components=2)
    # result_pca=[]
    # for i in range(0,data_len):
    #     pca_ = pca.fit_transform(result_norm[i])
    #     result_pca.append(pca_)
    data = []
    for i in range(num_cluster):
        data.append([])
    list_value = result_pca.tolist()
    for i in range(len(labels)):
        data[labels[i]].append(list_value[i])
    #그래프 그리기
    fig = go.Figure()
    data_np = np.array(data)
    
    for i in range(0,num_cluster):
        fig.add_trace(go.Scatter(
            x=[dt[0] for dt in data_np[i]], y= [dt[1] for dt in data_np[i]],
            mode='markers', name='Cluster'+str(i+1)
            )
        )
    graph = html.Div(style={}, children=[
        html.Div(["2-DIM VISUALIZATION"], className='subtitle'),
        html.Div(
            [html.Div(
                dcc.Graph(id='pca_show', figure=fig))
            ]
        )
    ])

    return graph