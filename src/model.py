import numpy as np 
import pandas as pd
from .network import Node, Edge, Network
from .splitter import Splitter
from .util import Util

class Model():
    def __init__(self, base_estimator, attributes=[]):
        self.base_estimator = base_estimator
        self.attributes = attributes
        self._network = Network()
    
    def fit(self, X, y=None):
        nodes = {Node(x) for _, x in X.iterrows()}
        self._network = Network(nodes)
        for attr in ['topics','title','abstract']:
                self._network._fit_nlp(attr)
                self._network._run_nlp(attr)
        E,y = Splitter.create_edges(nodes, 5.0)
        self._fit(E,y)
        return self
    
    def _fit(self, E, y, n_passes=0):
        X = self._transform(E)
        self.base_estimator.fit(X, y)
        if 0 < n_passes: 
            y_ = []
            X_ = []
            for x in E[:,1]:
                recs,_ = self._recommend(x, 5)
                X_.extend([[rec, x] for rec in recs])
                y_.extend([1 if np.any([rec,x] == X) else 0 for rec in recs])
            X_ = np.array([x for x in X_ if x in E[y.astype(bool)]])
            y_ = np.zeros(len(X_))
            X_,y_ = np.concatenate((E, X_)), np.append(y, y_)
            self._fit(X_, y_, n_passes-1)
        
    def predict(self, X):
        X_ = self._transform(X)
        return self.base_estimator.predict(X_)
            
    def predict_proba(self, X):
        X_ = self._transform(X)
        return self.base_estimator.predict_proba(X_)
         
    def _transform(self, E):
        X_ = np.zeros((E.shape[0], len(self.attributes)))
        for i, attr in enumerate(self.attributes):
            X_[:, i] = self._network.compare_many(E[:,0], E[:,1], attr)
        X_[np.isnan(X_)] = 0
        return X_
    
    def _update_network(self, X):
        if isinstance(X, pd.DataFrame):
            self._network.add_nodes(Node(x) for _, x in X.iterrows())
        elif isinstance(X, (np.ndarray, list, tuple)):
            self._network.add_nodes(Node(x) for x in X)
        for attr in self._network._tfidf:
            self._network._run_nlp(attr) 
        
    def _recommend(self, x, k=6):
        if isinstance(x, dict):
            x = Node.new_node(**x)
        if isinstance(x, str):
            x = self._network[x]
        E = np.array([[src, x.id] for src in self._network])
        self._network.add_nodes([x])
        pred = self.predict_proba(E)[:, 1]
        sort_idxs = np.argsort(pred)[::-1]
        return (E[sort_idxs][:k, 0], pred[sort_idxs][:k])
    
    def recommend(self, k=6, **kwargs):
        for i, (_id, prb) in enumerate(zip(*self._recommend(kwargs, k))):
            print(f'Match Probability: {prb*100:0.1f}%')
            print(self._network[_id].to_string())
            if i + 1 != k:
                usr_input = input('Any key for next suggestion or "exit" to quit: ')
            if "exit" in usr_input.lower():
                break
            
    def _page_ranks(self, v):
        adj_mat, idx = self._network.adj_matrix(idx=True)
        pg_rank = power_iteration(adj_mat, 50)
        idxs = np.array([idx[u] for u in v])
        return pg_rank[idxs].ravel()
         