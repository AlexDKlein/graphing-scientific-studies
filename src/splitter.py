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
    
    @staticmethod
    def create_edge_dataframe(X, in_='inCitations', out_='outCitations', directed=False, create_missing=False):
        edge_list = []
        nodes_set = set(X['_id'].values)
        for dst, srcs in X[in_].iteritems():
            for src in srcs: 
                if create_missing or (src in nodes_set and dst in nodes_set):
                    if not directed:
                        edge_list.append(sorted((src, str(dst))))
                    else:
                        edge_list.append((src, dst))
        if out_ is not None:
            for src, dsts in X[out_].iteritems():
                for dst in dsts: 
                    if create_missing or (src in nodes_set and dst in nodes_set):
                        edge_list.append([src, dst])
        edges = pd.DataFrame(edge_list, columns=['src', 'dst'])
        edges['id'] = edges.index
        edges['weight'] = 1
        edges = edges.groupby(['src','dst']).agg(
                {'id':'first',
                'weight': 'sum'}).reset_index()
        return edges


class EdgeSplitter():
    '''Wrapper class for node/edge split methods.
    ==============================================
    Usage: X0,E0,Xt,Et = EdgeSplitter().split(X,E)
    ==============================================
    '''
    def split(self, nodes, edges, n=10.0, **kwargs):
        if isinstance(n, float):
            n = int(n*len(edges))
        X0, Xt = train_test_split(nodes, **kwargs)
        E0, Et = [self.split_edges(X, edges).append(self.random_edges(X, n)) 
                for X in (X0, Xt)]
        return X0, Xt, E0, Et
    
    @staticmethod
    def transform(X, n=5.0, include_random=False, create_missing=False):
        E = EdgeSplitter.create_edges(X, create_missing=create_missing)
        E['edge'] = 1
        E = E.append(EdgeSplitter.random_edges(X, int(n*len(X))), sort=False)
        E.drop(['weight','id'], axis=1, inplace=True)
        return E

    @staticmethod
    def split_edges(nodes, edges):
        lst = []
        nodes_set = set(nodes['_id'].values)
        for src, dst in edges.loc[:, ['src','dst']].values:
            if src in nodes_set and dst in nodes_set:
                lst.append((src,dst, 1))
        return pd.DataFrame(lst,  columns=['src','dst','edge'])

    @staticmethod
    def random_edges(X, n=50000):
        lst = []
        for _ in range(n): 
            n1,n2 = sorted(np.random.choice(a=X['_id'], size=(2)))
            if n1!=n2 and [n1,n2]: lst.append([n1, n2,0])
        Z = pd.DataFrame(np.unique(lst,axis=0), columns=['src', 'dst', 'edge'])
        return Z

    @staticmethod
    def create_edges(X, in_='inCitations', out_='outCitations', directed=False, create_missing=False):
        edge_list = []
        nodes_set = set(X['_id'].values)
        for dst, srcs in X[in_].iteritems():
            for src in srcs: 
                if create_missing or (src in nodes_set and dst in nodes_set):
                    if not directed:
                        edge_list.append(sorted((src, str(dst))))
                    else:
                        edge_list.append((src, dst))
        if out_ is not None:
            for src, dsts in X[out_].iteritems():
                for dst in dsts: 
                    if create_missing or (src in nodes_set and dst in nodes_set):
                        edge_list.append([src, dst])
        edges = pd.DataFrame(edge_list, columns=['src', 'dst'])
        edges['id'] = edges.index
        edges['weight'] = 1
        edges = edges.groupby(['src','dst']).agg(
                {'id':'first',
                'weight': 'sum'}).reset_index()
        return edges
