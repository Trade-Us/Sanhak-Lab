# Clustering Algorithm
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, AgglomerativeClustering
from tslearn.clustering import TimeSeriesKMeans
# Needed Library!
import pandas as pd
import numpy as np

# Cluster Algorithm
def kmeans(dataset, n_clusters, n_init = 10, max_iter = 300, tol = e-4):

    cluster_data = KMeans(n_clusters=n_clusters, n_init = n_init, max_iter = max_iter, tol = tol).fit(dataset)
    return cluster_data


# DBSCAN
def dbscan(dataset, eps=0.5, min_samples=5):

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_data = dbscan.fit_predict(dataset)
    return cluster_data

# Spectral Clustering
def spectralClustering(dataset, n_clusters, n_init = 10):
    
    cluster_data = SpectralClustering(n_clusters=n_clusters, n_init=n_init).fit_predict(dataset)
    return cluster_data

# Hierarchical Clustering
def hierarchicalClustering(dataset, n_clusters, n_init = 10, linkage = 'ward'):

    cluster_data = AgglomerativeClustering(n_clusters = n_clusters, linkage = linkage, n_init=n_init ).fit(dataset)
    return cluster_data

 
def ts_kmeans_clustering(dataset, n_clusters, n_init=1, metric='euclidean'):
    cluster_data = TimeSeriesKMeans(n_clusters=n_clusters, n_init=n_init, metric=metric).fit(dataset)
    return cluster_data