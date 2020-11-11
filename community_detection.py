import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
import community as community_louvain

G = nx.erdos_renyi_graph(100,0.5)

def spectralPartitioning(G,n=None,**kwargs):
    mat = nx.to_numpy_matrix(G)
    if n is None:
        model = SpectralClustering(affinity="precomputed",**kwargs)
    else:
        model = SpectralClustering(n,affinity="precomputed",**kwargs)
    fit = model.fit()
    return {n:fit.labels_[i] for i,n in enumerate(G.nodes())}

def hierarchicalPartitioning(G,n,**kwargs):
    mat = nx.to_numpy_matrix(G)
    if n is None:
        model = AgglomerativeClustering(n,**kwargs)
    else:
        model = AgglomerativeClustering(**kwargs)
    fit = model.fit(mat)
    return {n:fit.labels_[i] for i,n in enumerate(G.nodes())}


def kMeansPartitioning(G,n,**kwargs):
    mat = nx.to_numpy_matrix(G)
    if n is None:
        model = KMeans(n,**kwargs)
    else:
        model = KMeans(**kwargs)
    fit = model.fit(mat)
    return {n:fit.labels_[i] for i,n in enumerate(G.nodes())}

def clausetNewmanMoorePartitioning(G):
    communities = nx.algorithms.community.greedy_modularity_communities(G)
    res = {}
    for i,com in enumerate(communities):
        for j in com:
            res[j] = i
    return res

def louvainParitioning(G):
    return community_louvain.best_partition(G)


