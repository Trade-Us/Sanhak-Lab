import pandas as pd
import numpy as np
from readFile import split_into_values, toRPdata
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc





# columns 와 value는 사용자 입력
df = pd.read_csv('data/CLAMP_resample.csv')
columns = ['chip', 'wire', 'segment']
value = ['value']
#df = pd.read_csv('resources/Dataset1.csv')
#columns = ['Process', 'Step']
#value = ['Value']

df = df.loc[:, columns + value] #('chip', 'wire', 'value')는 사용자 입력
result = split_into_values(df, columns)

min=result.dropna(axis='columns')
min_len=len(min.columns)
result_ = TimeSeriesResampler(sz=min_len).fit_transform(result)
result_=result_.reshape(1140,min_len)

result_norm = StandardScaler().fit_transform(result_)

pca = PCA(n_components=2) 
result_pca = pca.fit_transform(result_norm)

result_label=[1,2,3,4,5,6]*190
result_list=result_pca.tolist()
result_cluster=[[],[],[],[],[],[]]

for i in range(0,1140):
    if result_label[i] ==1:
        result_cluster[0].append(result_list[i])
    elif result_label[i] ==2:
        result_cluster[1].append(result_list[i])
    elif result_label[i] ==3:
        result_cluster[2].append(result_list[i])
    elif result_label[i] ==4:
        result_cluster[3].append(result_list[i])
    elif result_label[i] ==5:
        result_cluster[4].append(result_list[i])
    elif result_label[i] ==6:
        result_cluster[5].append(result_list[i])

result_arr = np.array(result_cluster)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=result_arr[0][:,0], y= result_arr[0][:,1],
    mode='markers', name='cluster1'
    )
)
fig.add_trace(go.Scatter(
    x=result_arr[1][:,0], y= result_arr[1][:,1],
    mode='markers', name='cluster2'
    )
)
fig.add_trace(go.Scatter(
    x=result_arr[2][:,0], y= result_arr[2][:,1],
    mode='markers', name='cluster3'
    )
)
fig.add_trace(go.Scatter(
    x=result_arr[3][:,0], y= result_arr[3][:,1],
    mode='markers', name='cluster4'
    )
)
fig.add_trace(go.Scatter(
    x=result_arr[4][:,0], y= result_arr[4][:,1],
    mode='markers', name='cluster5'
    )
)
fig.add_trace(go.Scatter(
    x=result_arr[5][:,0], y= result_arr[5][:,1],
    mode='markers', name='cluster6'
    )
)


def pca_show():
    graph = html.Div(style={}, children=[
        html.H4("군집 2차원 시각화"),
        html.Div(
            [html.Div(
                dcc.Graph(id='pca_show', figure=fig))
            ]
        )
    ])

    return graph