import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

def unit_vect(shape):
    v = np.random.rand(*shape)
    v /= np.linalg.norm(v)
    return v

def pagerank(X, eps=1e-3, p=0.85):
    X_ = (p * X) + (1 - p) / X.shape[1]
    v, v_ = unit_vect((X.shape[1], 1)), 1
    while np.linalg.norm(v - v_) > eps:
        v_ = v
        v = X_ @ v / np.linalg.norm(v)
    return v

def power_iteration(X, n):
    v = unit_vect(X.shape[1])
    for _ in range(n):
        v_ = X @ v
        v = v_ / np.linalg.norm(v)
    return v

def edge_matrix(E):
    pass
    
def simrank(S, A, C=0.5):
    return C*(A.T @ S @ A) + np.identity(A.shape[0])

def adj_matrix(edges, nodelist=None):
    g = nx.Graph(edges)
    return nx.adj_matrix(g,nodelist=nodelist)

class Graph():
    def __init__(self, edges={}, idxs=None):
        self.edges = edges
        self.idxs = idxs if idxs is not None else np.arange(stop=0)

    def __matmul__(self, other):
        pass