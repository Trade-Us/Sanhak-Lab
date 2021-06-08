import dash_html_components as html

from cluster_layout import *
from dim_rduction_layout import *
from img_layout import *
from normalize_layout import *

def ts_sample_kmeans():
    return html.Div(id='ts-sample-kmeans', children=[
        html.Div(id='hidden-ts-sample-kmeans', style={'display':'none'}),
        timeseries_resampler_layout(),
        kmeans_layout()
    ])
def ts_sample_hierarchy():
    return html.Div(id='ts-sample-hierarchy', children=[
        html.Div(id='hidden-ts-sample-hierarchy', style={'display':'none'}),
        timeseries_resampler_layout(),
        hierarchy_layout()
    ])
def ts_sample_dbscan():
    return html.Div(id='ts-sample-dbscan', children=[
        html.Div(id='hidden-ts-sample-dbscan', style={'display':'none'}),
        timeseries_resampler_layout(),
        dbscan_layout()
    ])
def ts_sample_ts_kmeans():
    return html.Div(id='ts-sample-kmeans', children=[
        html.Div(id='hidden-ts-sample-ts-kmeans', style={'display':'none'}),
        timeseries_resampler_layout(),
        time_sereies_kmeans_layout()
    ])
def rp_ae_kmeans():
    return html.Div(id='rp-ae-kmeans', children=[
        html.Div(id='hidden-rp-ae-kmeans', style={'display':'none'}),
        rp_layout(),
        autoencoder_layout(),
        kmeans_layout(),
    ])
def rp_ae_hierarchy():
    return html.Div(id='rp-ae-hierarchy', children=[
        html.Div(id='hidden-rp-ae-hierarchy', style={'display':'none'}),
        rp_layout(),
        autoencoder_layout(),
        hierarchy_layout()
    ])
def rp_ae_dbscan():
    return html.Div(id='rp-ae-dbscan', children=[
        html.Div(id='hidden-rp-ae-dbscan', style={'display':'none'}),
        rp_layout(),
        autoencoder_layout(),
        dbscan_layout()
    ])

def gaf_ae_kmeans():
    return html.Div(id='gaf-ae-kmeans', children=[
        html.Div(id='hidden-gaf-ae-kmeans', style={'display':'none'}),
        gaf_layout(),
        autoencoder_layout(),
        kmeans_layout(),
    ])
def gaf_ae_hierarchy():
    return html.Div(id='gaf-ae-hierarchy', children=[
        html.Div(id='hidden-gaf-ae-hierarchy', style={'display':'none'}),
        gaf_layout(),
        autoencoder_layout(),
        hierarchy_layout()
    ])
def gaf_ae_dbscan():
    return html.Div(id='gaf-ae-dbscan', children=[
        html.Div(id='hidden-gaf-ae-dbscan', style={'display':'none'}),
        gaf_layout(),
        autoencoder_layout(),
        dbscan_layout()
    ])

def mtf_ae_kmeans():
    return html.Div(id='mtf-ae-kmeans', children=[
        html.Div(id='hidden-mtf-ae-kmeans', style={'display':'none'}),
        mtf_layout(),
        autoencoder_layout(),
        kmeans_layout(),
    ])
def mtf_ae_hierarchy():
    return html.Div(id='mtf-ae-hierarchy', children=[
        html.Div(id='hidden-mtf-ae-hierarchy', style={'display':'none'}),
        mtf_layout(),
        autoencoder_layout(),
        hierarchy_layout()
    ])
def mtf_ae_dbscan():
    return html.Div(id='mtf-ae-dbscan', children=[
        html.Div(id='hidden-mtf-ae-dbscan', style={'display':'none'}),
        mtf_layout(),
        autoencoder_layout(),
        dbscan_layout()
    ])

def wavelet_kmeans():
    return html.Div(id='wavelet-kmeans', children=[
        html.Div(id='hidden-wavelet-kmeans', style={'display':'none'}),
        wavelet_layout(),
        kmeans_layout()
    ])
def wavelet_hierarchy():
    return html.Div(id='wavelet-hierarchy', children=[
        html.Div(id='hidden-wavelet-hierarchy', style={'display':'none'}),
        wavelet_layout(),
        dbscan_layout()
    ])
def wavelet_dbscan():
    return html.Div(id='wavelet-dbscan', children=[
        html.Div(id='hidden-wavelet-dbscan', style={'display':'none'}),
        wavelet_layout(),
        hierarchy_layout()
    ])