import numpy as np 
import pandas as pd
from .network import Node, Edge, Network
from sklearn.model_selection import train_test_split

class Splitter():
    @staticmethod
    def split(X, random=0.0, **kwargs):
        X0, Xt = train_test_split(X, **kwargs)
        E0,y0 = Splitter.create_edges(X0, random=random)
        Et,yt = Splitter.create_edges(Xt, random=random)
        return X0,Xt, y0,yt, E0,Et
        
    @staticmethod
    def create_edges(X, random=0.0):
        if isinstance(X, pd.DataFrame):
            X = {Node(x) for _,x in X.iterrows()}
        return Splitter._construct_edges(X, int(random * len(X)))  
        
    @staticmethod
    def _construct_edges(nodes, n_random):
        edges = set()
        node_ids = {node.id for node in nodes}
        
        src_idxs=np.random.choice(np.arange(len(nodes)), n_random)
        dst_idxs=np.random.choice(np.arange(len(nodes)), n_random)

        rand_srcs = np.array(['' for _ in range(n_random)], dtype='O')
        rand_dsts = np.array(['' for _ in range(n_random)], dtype='O')
        for i, node in enumerate(nodes):
            src_match = (src_idxs == i)
            dst_match = (dst_idxs == i)
            if np.any(src_match):
                rand_srcs[src_match] = node.id
            if np.any(dst_match):
                rand_dsts[dst_match] = node.id
            for src in node.in_:
                if src in node_ids:
                    edges |= {(src, node.id)} 
            for dst in node.out_:
                if dst in node_ids:
                    edges |= {(node.id, dst)}
                    
        rand_E = np.array(list(zip(rand_srcs, rand_dsts)))
        E = np.array(list(edges))
        y = np.ones(len(E))
        rand_y = np.zeros(len(rand_E))
        E = np.append(E, rand_E, axis=0)
        y = np.append(y, rand_y, axis=0)
        return E,y
    