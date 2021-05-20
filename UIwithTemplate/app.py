import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd

# Import for algorithm
# from utils.readFile import*
# from utils.dim_reduction import*
from utils.readFile import split_into_values
from utils.dim_reduction import exec_ts_resampler, fit_autoencoder
from utils.normalize import MinMax, Standard,Robust,MaxAbsScaler,tsleanr_scaler
from utils.clustering import *
from utils.imagination import *
from utils.metric import *
from clusters import *
from read_csv import csvDiv, parse_contents
from result_graph import graphDetail, graphCluster, graphBig, textResultDiv
import show_detail as sd

from visualization import pca_show

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    suppress_callback_exceptions=True
)
server = app.server


# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("dash-logo.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H1(
                                    "Time-series Clustering ",
                                ),
                                html.H5(
                                    "Kwangwoon Univ. team 일이삼사", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.Button("학습 시작하기", id="learn-button")
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        # 파라미터 조작, 파일삽입, 군집화 결과 컴포넌트 틀
        html.Div(
            [
                # 파라미터 조작 컴포넌트
                html.Div([
                    html.H3("Parameters"),
                    dcc.Dropdown(
                        id='main-cluster-algorithm',
                        options=[
                                    {'label': 'TimeSeriesSample + KMeans', 'value':'ts_sample_kmeans'},
                                    {'label': 'TimeSeriesSample + Hierarchical Cluster', 'value':'ts_sample_hierarchy'},
                                    {'label': 'TimeSeriesSample + TimeSeriesKMeans', 'value':'ts_sample_ts_kmeans'},
                                    {'label': 'RP + Autoencoder + Kmeans', 'value':'rp_ae_kmeans'},
                                    {'label': 'RP + Autoencoder + Hierarchical Cluster', 'value':'rp_ae_hierarchy'},
                                    {'label': 'RP + Autoencoder + DBSCAN', 'value':'rp_ae_dbscan'},
                                    {'label': 'GAF + Autoencoder + Kmeans', 'value':'gaf_ae_kmeans'},
                                    {'label': 'GAF + Autoencoder + Hierarchical Cluster', 'value':'gaf_ae_hierarchy'},
                                    {'label': 'GAF + Autoencoder + DBSCAN', 'value':'gaf_ae_dbscan'},
                                    {'label': 'MTF + Autoencoder + Kmeans', 'value':'mtf_ae_kmeans'},
                                    {'label': 'MTF + Autoencoder + Hierarchical Cluster', 'value':'mtf_ae_hierarchy'},
                                    {'label': 'MTF + Autoencoder + DBSCAN', 'value':'mtf_ae_dbscan'},
                                    {'label': 'Wavelet + Kmeans', 'value':'wavelet_kmeans'},
                                    {'label': 'Wavelet + Hierarchical Cluster', 'value':'wavelet_hierarchy'},
                                    {'label': 'Wavelet + DBSCAN', 'value':'wavelet_dbscan'},
                                ],
                        value='ts_sample_kmeans'),
                    html.Div(id='hidden-columns', style={'display':'none'}),
                    html.Hr(),
                    html.H4('csv file Attributes'),
                    html.H6('시계열 데이터 셋을 구분하는 column을 설정해주세요. (1개 이상)'),
                    dcc.Dropdown(
                        id='partitioning-column-data',
                        multi=True
                    ),
                    html.H6('데이터의 값을 나타내는 column을 설정해주세요. (1개)'),
                    dcc.Dropdown(
                        id='main-ts-data'
                    ),
                    html.Hr(),
                    html.Div(id='normalization-param', children=[
                    dcc.Store(id='store-normalization-param', data=[]),
                    html.H4("Normalization Parameters"),
                    html.H6('Normalization method Before Training'),
                    dcc.Dropdown(id='normalization-method',
                        options=[
                            {'label': 'MinMax Scaler', 'value':'MMS'},
                            {'label': 'Standard Scaler', 'value':'SSC'},
                            {'label': 'Robust Scaler', 'value':'RBS'},
                            {'label': 'Max Abs Scaler', 'value':'MAS'},
                            {'label': 'Time Series Scaler Mean Variance', 'value':'TSS'}
                        ], value='MMS'),
                ]),
                        html.Hr(),
                    html.Div(id='parameter-layout')],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                # 오른쪽 부분 컴포넌트 틀 (파일 삽입, 군집화 결과)
                html.Div(
                    [
                        # 파일 삽입 컴포넌트
                        html.Div(
                            [
                                html.Div(
                                    [csvDiv()],
                                    className="mini_container full",
                                )
                            ],
                            id="info-container",
                            className="row container-display",
                        ),

                        # 군집화 결과 그래프 컴포넌트
                        html.Div([
                            html.Div([html.Div("CLUSTERING REPORT")], className='textTitle'),
                            html.Hr(),
                            html.Div([
                                html.Div(id='text-result',
                                    # textResultDiv(), pca_show(), graphCluster(), sd.detailGraphOption(),
                                className='text-pca1 textdiv'),
                                html.Div(id='graph-cluster-result',
                                    # pca_show()
                                className='text-pca2')
                            ],  className='row container-display'),
                            html.Div(id='graph-result',
                                # graphCluster()
                            className = 'box-scroll')
                        ],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
        # 하단/ 군집화 세부적 결과 그래프 (크게보기, 하나씩 보기)
        html.Div(
            [
                html.Div(['Graph View'],className='subtitle'),
                # 세부적 결과 그래프 컴포넌트
                html.Div(
                    id='detail-graph-output'
                ),
                html.Div(id='detail-graph-option',
                    # sd.detailGraphOption(),
                    className = "floatright"
                )
            ],
            className="pretty_container row flex-display marginleft",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


##read_csv
@app.callback(Output('output-data-upload', 'children'),
              Output('partitioning-column-data', 'options'),
              Output('main-ts-data', 'options'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              prevent_initial_call=True 
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        result = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return result[0][0], result[0][1], result[0][1]

####                                                           ####
# 컨트롤 컴포넌트에 의해 세부적 그래프 컴포넌트가 달라집니다. #
####                                                           ####
@app.callback(
    # my-output id를 가진 컴포넌트의 children 속성으로 들어간다.
    Output(component_id='detail-graph-output', component_property='children'),
    Output(component_id='detail-graph-output', component_property='className'),
    Output(component_id='num-of-graphs', component_property='max'),
    Output(component_id='num-of-graphs', component_property='value'),
    Output(component_id='label-n-graphs', component_property='children'),
    # my-input id 를 가진 컴포넌트의 value 속성을 가져온다.
    # Input('detail-graph-submit', 'n_clicks'),
    Input(component_id='nth-cluster', component_property='value'),
    Input(component_id='detail-graph-input', component_property='value'),
    Input(component_id='num-of-graphs', component_property='value')
)
def update_parameter( nth_cluster, detail_graph, num_graph):
    layout = []
    clsName = ''
    nMaxGraphs = len(GG[nth_cluster])
    if num_graph is None or num_graph > nMaxGraphs:
        num_graph = nMaxGraphs
    if detail_graph == 'GrDt':
        layout = graphDetail(nth_cluster, num_graph, GG)
        clsName = "box-scroll"
    elif detail_graph == 'GrBg':
        layout = graphBig(nth_cluster, num_graph, GG)
        clsName = "fullgraph_class"
    #최대 그래프 개수

    return layout, clsName, nMaxGraphs, num_graph, f"Number of data graphs per clusters (max: {nMaxGraphs})"
#######################################################################
@app.callback(
    Output('parameter-layout', 'children'),
    Input('main-cluster-algorithm', 'value')
)
def select_main_algorithm(algorithm):
    if algorithm == 'ts_sample_kmeans':
        return ts_sample_kmeans()
    elif algorithm == 'ts_sample_hierarchy':
        return ts_sample_hierarchy()
    elif algorithm == 'ts_sample_ts_kmeans':
        return ts_sample_ts_kmeans()
    elif algorithm == 'rp_ae_kmeans':
        return rp_ae_kmeans()
    elif algorithm == 'rp_ae_hierarchy':
        return rp_ae_hierarchy()
    elif algorithm == 'rp_ae_dbscan':
        return rp_ae_dbscan()
    elif algorithm == 'gaf_ae_kmeans':
        return gaf_ae_kmeans()
    elif algorithm == 'gaf_ae_hierarchy':
        return gaf_ae_hierarchy()
    elif algorithm == 'gaf_ae_dbscan':
        return gaf_ae_dbscan()
    elif algorithm == 'mtf_ae_kmeans':
        return mtf_ae_kmeans()
    elif algorithm == 'mtf_ae_hierarchy':
        return mtf_ae_hierarchy()
    elif algorithm == 'mtf_ae_dbscan':
        return mtf_ae_dbscan()
    elif algorithm == 'wavelet_kmeans':
        return wavelet_kmeans()
    elif algorithm == 'wavelet_hierarchy':
        return wavelet_hierarchy()
    elif algorithm == 'wavelet_dbscan':
        return wavelet_dbscan()
#######################################################################
#  각 알고리즘 별 변수 저장
# KMeans 관련 parameter
@app.callback(
    Output('store-kmeans-param', 'data'),
    Input("number-of-cluster", "value"),
    Input("tolerance", "value"),
    Input("try-n-init", "value"),
    Input("try-n-kmeans", "value"),
    Input("random-center", "value"),
)
def store_kmeans_param(ncl, tol, tni, tnk, rc):
    df = pd.DataFrame()
    df['number_of_cluster'] = [ncl]
    df['tolerance'] = [tol]
    df['try_n_init'] = [tni]
    df['try_n_kmeans'] = [tnk]
    df['random_center'] = [rc]
    data = df.to_dict('records')
    return data
# hirarchy cluster 관련 parameter
@app.callback(
    Output('store-hierarchy-param', 'data'),
    Input("number-of-cluster", "value"),
    Input("try-n-init", "value"),
    Input("linkage", "value"),
)
def store_hirarchy_param(ncl, tni, lnk):
    df = pd.DataFrame()
    df['number_of_cluster'] = [ncl]
    df['try_n_init'] = [tni]
    df['linkage'] = [lnk]
    data = df.to_dict('records')
    return data
# DBSCAN 관련 parameter
@app.callback(
    Output('store-dbscan-param', 'data'),
    Input("dbscan-epsilon", "value"),
    Input("dbscan-min-sample", "value")
)
def store_dbscan_param(eps, msp):
    df = pd.DataFrame()
    df['epsilon'] = [eps]
    df['min_sample'] = [msp]
    data = df.to_dict('records')
    return data

# Image Data(RP) 관련 Parameter
@app.callback(
    Output('store-gaf-param', 'data'),
    Input("image-size", "value"),
    Input("gaf-method", "value")
)
def store_gaf_param(img_size, method):
    df = pd.DataFrame()
    df['image_size'] = [img_size]
    df['gaf_method'] = [method]
    data = df.to_dict('records')
    return data

@app.callback(
    Output('store-mtf-param', 'data'),
    Input("image-size", "value"),
    Input("mtf-n-bins", "value"),
    Input("mtf-strategy", "value"),
)
def store_mtf_param(img_size, n_bins, strategy):
    df = pd.DataFrame()
    df['image_size'] = [img_size]
    df['n_bins'] = [n_bins]
    df['mtf_strategy'] = [strategy]
    data = df.to_dict('records')
    return data

@app.callback(
    Output('store-rp-param', 'data'),
    Input("image-size", "value"),
    Input("dimension", "value"),
    Input("time-delay", "value"),
    Input("threshold", "value"),
    Input("percentage", "value"),
)
def store_rp_param(img_size, dim, td, th, prtg):
    df = pd.DataFrame()
    df['image_size'] = [img_size]
    df['dimension'] = [dim]
    df['time_delay'] = [td]
    df['threshold'] = [th]
    df['percentage'] = [prtg]
    data = df.to_dict('records')
    return data

@app.callback(
    Output('store-autoencoder-param', 'data'),
    Input("autoencoder-batch-size", "value"),
    Input("autoencoder-learning-rate", "value"),
    Input("autoencoder-loss-function", "value"),
    Input("autoencoder-activation-function", "value"),
    Input("autoencoder-epoch", "value"),
    Input("autoencoder-optimizer", "value"),
    Input("autoencoder-dimension-feautre", "value"),
)
def store_ae_param(bs, lr, loss_f, act_f, epoch, optm, dim):
    df = pd.DataFrame()
    df['batch_size'] = [bs]
    df['learning_rate'] = [lr]
    df['loss_function'] = [loss_f]
    df['activation_function'] = [act_f]
    df['epoch'] = [epoch]
    df['optimizer'] = [optm]
    df['dimension_feature'] = [dim]
    data = df.to_dict('records')
    return data

@app.callback(
    Output('store-wavelet-param', 'data'),
    Input("wavelet-function", "value"),
    Input("iteration-make-half-dim", "value")
)
def store_wavelet_param(func, iter):
    df = pd.DataFrame()
    df['wavelet_func'] = [func]
    df['iter_to_half'] = [iter]
    data = df.to_dict('records')
    return data
# Autoencoder (ae) 관련 Parameter
@app.callback(
    Output('store-ts-resampler-param', 'data'),
    Input("ts-resampler-dim", "value")
)
def store_ts_resampler_param(dim):
    df = pd.DataFrame()
    df['dimension'] = [dim]
    data = df.to_dict('records')
    return data
@app.callback(
    Output('store-normalization-param', 'data'),
    Input('normalization-method', 'value'),
)
def store_normalization_param(method):
    df = pd.DataFrame()
    df['normalization'] = [method]
    data = df.to_dict('records')
    return data
#######################################################################
## 군집화 알고리즘 별 파라미터 호출
# timeSeriesSample + kmeans
@app.callback(
    Output("hidden-ts-sample-kmeans", "children"),
    Input("learn-button", "n_clicks"),
    State("store-kmeans-param", 'data'),
    State("store-ts-resampler-param", "data"),
    State("partitioning-column-data", 'value'),
    State('main-ts-data', 'value'),
    State('normalization-method', 'value'),
    prevent_initial_call=True 
)
def exct_ts_sample_kmeans(n_clicks, km_data, tsre_data, parti_columns, value, normalize):
    print(km_data)
    print(tsre_data)
    print(parti_columns)
    print(value)
    print(normalize)

    df = pd.read_csv('.\\data\\saved_data.csv')
    value_ = [value]
    df = df.loc[:,parti_columns + value_]
    result = split_into_values(df, parti_columns)
    min=result.dropna(axis='columns')
    min_len = len(min.columns) if tsre_data[0]['dimension'] is None else tsre_data[0]['dimension']
    print(min_len)
    if normalize == "MMS":
        result_nom = MinMax(result)
    result_ = exec_ts_resampler(result_nom,min_len)
    result_ = result_.reshape(result_.shape[0],min_len)
    print(result_.shape)
    
    # print(result.labels_)

    cluster = kmeans(result_,km_data[0]['number_of_cluster'] , km_data[0]['tolerance'],km_data[0]['try_n_init'],km_data[0]['try_n_kmeans'])

    global num_cluster, num_tsdatas_per_cluster, siluet_score, used_algorithm, labels, GG, origin_data, execution
    origin_data = []
    # 결과 데이터
    GG = []
    # 클러스터 개수
    num_cluster = 0
    # 클러스터 당 시계열 데이터 개수
    num_tsdatas_per_cluster = []
    # 실루엣 점수
    siluet_score = 0
    # 사용 알고리즘
    used_algorithm = ''
    labels = []
    origin_data = result
    num_cluster = km_data[0]['number_of_cluster']
    used_algorithm = 'Time Series Resampler & Kmeans'# 사용한 알고리즘 적용
    siluet_score = plotSilhouette(result_ ,cluster)
    labels = cluster.labels_
    print(labels)
    for i in range(num_cluster):
        GG.append([])
    list_value = result.values.tolist()
    for i in range(len(labels)):
        GG[labels[i]].append(list_value[i])
    num_tsdatas_per_cluster = [len(ts_data) for ts_data in GG]
    execution = True
    return 0
# timeSeriesSample + hierarchy
@app.callback(
    Output("hidden-ts-sample-hierarchy", "children"),
    Input("learn-button", "n_clicks"),
    State("store-hierarchy-param", 'data'),
    prevent_initial_call=True 
)
def exct_ts_sample_hierarchy(n_clicks, hrc_data):
    print(hrc_data)
    return []
# rp-ae-kmeans
@app.callback(
    Output("hidden-rp-ae-kmeans", 'children'),
    Input("learn-button", "n_clicks"),
    State("store-rp-param", 'data'),
    State("store-autoencoder-param", 'data'),
    State("store-kmeans-param", 'data'),
    State("partitioning-column-data", 'value'),
    State('main-ts-data', 'value'),
    State('normalization-method', 'value'),
    prevent_initial_call=True 
)
def exct_rp_autoencoder_kmeans(n_clicks, rp_data, ae_data, km_data,  parti_columns, value, normalize):
    print(rp_data)
    print(ae_data)
    print(km_data)
    print(parti_columns)
    print(value)
    print(normalize)
    threshold = rp_data[0]['threshold']
    if threshold == 'None':
        threshold = None

    df = pd.read_csv('.\\data\\saved_data.csv')
    value_ = [value]
    df = df.loc[:,parti_columns + value_]
    result = split_into_values(df, parti_columns)
    min=result.dropna(axis='columns')
    min_len=len(min.columns)
    if normalize == "MMS":
        result_nom = MinMax(result)
    result_resample = exec_ts_resampler(result_nom,rp_data[0]['image_size'])
    #(242,28,1)
    result_ = result_resample.reshape(result_resample.shape[0],1,result_resample.shape[1])
    #(242,28,28)
    X = toRPdata(result_,rp_data[0]['dimension'],rp_data[0]['time_delay'],threshold,rp_data[0]['percentage'] / 100)
    X_expand = np.expand_dims(X,axis=3)

    all_feature = fit_autoencoder(X_expand,rp_data[0]['image_size'],ae_data[0]['dimension_feature'],ae_data[0]['optimizer'],(3e-7) * (10**ae_data[0]['learning_rate']),ae_data[0]['activation_function'],ae_data[0]['loss_function'],ae_data[0]['batch_size'],ae_data[0]['epoch'])
    print(f'feature shape{all_feature.shape}')
    cluster = kmeans(all_feature, km_data[0]['number_of_cluster'] , km_data[0]['tolerance'], km_data[0]['try_n_init'], km_data[0]['try_n_kmeans'])

    global num_cluster, num_tsdatas_per_cluster, siluet_score, used_algorithm, labels, GG, origin_data, execution
    origin_data = []
    # 결과 데이터
    GG = []
    # 클러스터 개수
    num_cluster = 0
    # 클러스터 당 시계열 데이터 개수
    num_tsdatas_per_cluster = []
    # 실루엣 점수
    siluet_score = 0
    # 사용 알고리즘
    used_algorithm = ''
    labels = []
    origin_data = result
    num_cluster = km_data[0]['number_of_cluster']
    used_algorithm = 'RP Autoencoder & Kmeans'# 사용한 알고리즘 적용
    siluet_score = plotSilhouette(result_resample.reshape(result_resample.shape[0], rp_data[0]['image_size']) ,cluster)
    labels = cluster.labels_
    print(labels)
    for i in range(num_cluster):
        GG.append([])
    list_value = result.values.tolist()
    for i in range(len(labels)):
        GG[labels[i]].append(list_value[i])
    num_tsdatas_per_cluster = [len(ts_data) for ts_data in GG]
    execution = True
    return 0
# rp-ae-hierarchy
@app.callback(
    Output("hidden-rp-ae-hierarchy", "children"),
    Input("learn-button", "n_clicks"),
    State("store-rp-param", 'data'),
    State("store-autoencoder-param", 'data'),
    State("store-hierarchy-param", 'data'),
    prevent_initial_call=True 
)
def exct_rp_autoencoder_hierarchy(n_clicks, rp_data, ae_data, hrc_data):
    print(rp_data)
    print(ae_data)
    print(hrc_data)
    return []
# rp-ae-dbscan
@app.callback(
    Output("hidden-rp-ae-dbscan", "children"),
    Input("learn-button", "n_clicks"),
    State("store-rp-param", 'data'),
    State("store-autoencoder-param", 'data'),
    State("store-dbscan-param", 'data'),
    prevent_initial_call=True 
)
def exct_rp_autoencoder_dbscan(n_clicks, rp_data, ae_data, dbs_data):
    print(rp_data)
    print(ae_data)
    print(dbs_data)
    return []

# gaf-ae-kmeans
@app.callback(
    Output("hidden-gaf-ae-kmeans", "children"),
    Input("learn-button", "n_clicks"),
    State("store-gaf-param", 'data'),
    State("store-autoencoder-param", 'data'),
    State("store-kmeans-param", 'data'),
    prevent_initial_call=True 
)
def exct_gaf_autoencoder_kmeans(n_clicks, gaf_data, ae_data, km_data):
    print(gaf_data)
    print(ae_data)
    print(km_data)
    return []
# gaf-ae-hierarchy
@app.callback(
    Output("hidden-gaf-ae-hierarchy", "children"),
    Input("learn-button", "n_clicks"),
    State("store-gaf-param", 'data'),
    State("store-autoencoder-param", 'data'),
    State("store-hierarchy-param", 'data'),
    prevent_initial_call=True 
)
def exct_gaf_autoencoder_hierarchy(n_clicks, gaf_data, ae_data, hrc_data):
    print(gaf_data)
    print(ae_data)
    print(hrc_data)
    return []
# gaf-ae-dbscan
@app.callback(
    Output("hidden-gaf-ae-dbscan", "children"),
    Input("learn-button", "n_clicks"),
    State("store-gaf-param", 'data'),
    State("store-autoencoder-param", 'data'),
    State("store-dbscan-param", 'data'),
    prevent_initial_call=True 
)
def exct_gaf_autoencoder_dbscan(n_clicks, gaf_data, ae_data, dbs_data):
    print(gaf_data)
    print(ae_data)
    print(dbs_data)
    return []

# mtf-ae-kmeans
@app.callback(
    Output("hidden-mtf-ae-kmeans", "children"),
    Input("learn-button", "n_clicks"),
    State("store-mtf-param", 'data'),
    State("store-autoencoder-param", 'data'),
    State("store-kmeans-param", 'data'),
    prevent_initial_call=True 
)
def exct_mtf_autoencoder_kmeans(n_clicks, mtf_data, ae_data, km_data):
    print(mtf_data)
    print(ae_data)
    print(km_data)
    return []
# mtf-ae-hierarchy
@app.callback(
    Output("hidden-mtf-ae-hierarchy", "children"),
    Input("learn-button", "n_clicks"),
    State("store-mtf-param", 'data'),
    State("store-autoencoder-param", 'data'),
    State("store-hierarchy-param", 'data'),
    prevent_initial_call=True 
)
def exct_mtf_autoencoder_hierarchy(n_clicks, mtf_data, ae_data, hrc_data):
    print(mtf_data)
    print(ae_data)
    print(hrc_data)
    return []
# mtf-ae-dbscan
@app.callback(
    Output("hidden-mtf-ae-dbscan", "children"),
    Input("learn-button", "n_clicks"),
    State("store-mtf-param", 'data'),
    State("store-autoencoder-param", 'data'),
    State("store-dbscan-param", 'data'),
    prevent_initial_call=True 
)
def exct_mtf_autoencoder_dbscan(n_clicks, mtf_data, ae_data, dbs_data):
    print(mtf_data)
    print(ae_data)
    print(dbs_data)
    return []
# wavelet-kmeans
@app.callback(
    Output("hidden-wavelet-kmeans", "children"),
    Input("learn-button", "n_clicks"),
    State("store-wavelet-param", 'data'),
    State("store-kmeans-param", 'data'),
    prevent_initial_call=True 
)
def exct_wavelet_kmeans(n_clicks, wav_data, kms_data):
    print(wav_data)
    print(kms_data)
    return []
# wavelet-hierarchy
@app.callback(
    Output("hidden-wavelet-hierarchy", "children"),
    Input("learn-button", "n_clicks"),
    State("store-wavelet-param", 'data'),
    State("store-hierarchy-param", 'data'),
    prevent_initial_call=True 
)
def exct_wavelet_hierarchy(n_clicks, wav_data, hrc_data):
    print(wav_data)
    print(hrc_data)
    return []
# wavelet-dbscan
@app.callback(
    Output("hidden-wavelet-dbscan", "children"),
    Input("learn-button", "n_clicks"),
    State("store-wavelet-param", 'data'),
    State("store-dbscan-param", 'data'),
    prevent_initial_call=True 
)
def exct_wavelet_dbscan(n_clicks, wav_data, dbs_data):
    print(wav_data)
    print(dbs_data)
    return []
import time
@app.callback(
    Output("text-result", 'children'),
    Output("graph-cluster-result", 'children'),
    Output("graph-result", 'children'),
    Output("detail-graph-option", 'children'),
    Input("learn-button", "n_clicks"),
    # input
    prevent_initial_call=True 
)
def show_result1(change):

    global execution, total_time
    while not execution:
        print('실행 중...')
        time.sleep(2)
        total_time += 2
    print("완료")
    time_ = total_time
    total_time = 0
    execution = False
    return textResultDiv(num_cluster, num_tsdatas_per_cluster, siluet_score, used_algorithm, time_),\
    pca_show(origin_data, labels, num_cluster),\
    graphCluster(GG),\
    sd.detailGraphOption(num_cluster)
# 학습 버튼을 클릭 하게 되면, i
# Main
if __name__ == "__main__":
    execution = False
    # 본 데이터
    origin_data = []
    # 결과 데이터
    GG = []
    # 클러스터 개수
    num_cluster = 0
    # 클러스터 당 시계열 데이터 개수
    num_tsdatas_per_cluster = []
    # 실루엣 점수
    siluet_score = 0
    # 사용 알고리즘
    used_algorithm = ''
    labels = []
    total_time = 0
    app.run_server(debug=True, threaded=True)