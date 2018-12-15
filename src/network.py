import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix
from .util import Util

class Node():
    def __init__(self, x):
        self.id = x['_id'] if '_id' in x else '-1'
        self.authors = (set((itm['ids'][0] if itm['ids'] else -1) 
                            if isinstance(itm, dict) else itm
                            for itm in x['authors'])
                        if 'authors' in x else set())
        self.in_ = set(x['inCitations']) if 'inCitations' in x else set()
        self.out_ = set(x['outCitations']) if 'outCitations' in x else set()
        self.year = x['year'] if 'inCitations' in x else 2018
        self.title = x['title'] if 'title' in x else ''
        self.abstract = x['paperAbstract'] if 'paperAbstract' in x else ''
        self.topics = set(x['entities']) if 'entities' in x else set()
        
    def __repr__(self):
        return f'<{self.id}>'
    
    def to_string(self):
        return f'''
            {self.title}
            {str(self.year).center(len(self.title))}
            
        {self.abstract}
        
        Tags: {', '.join(self.topics)}
        '''
    
    def to_dict(self):
        return {
            'id':self.id,
            'authors': self.authors,
            'inCitations': self.in_,
            'outCitations': self.out_,
            'year': self.year,
            'title': self.title,
            'paperAbstract': self.abstract,
            'entities': self.topics
        }
        
    @property
    def n_authors(self):
        return len(self.authors)
    
    @staticmethod
    def new_node(**kwargs):
        return Node(kwargs)
        
class Edge():
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        
    def __repr__(self):
        return f'Edge(src={self.src}, dst={self.dst})'
    
    def similarity(self, attr):
        src_val = getattr(self.src, attr)
        dst_val = getattr(self.dst, attr)
        src_val = src_val if isinstance(src_val, str) else ' '.join(src_val)
        dst_val = dst_val if isinstance(dst_val, str) else ' '.join(dst_val)
        if len(src_val) == 0 or len(dst_val) == 0:
            return 0
        tfidf = TfidfVectorizer()
        tfidf.fit([src_val, dst_val])
        u,v = tfidf.transform([src_val, dst_val])
        similarity = (u @ v.T)
        return similarity.data[0] if similarity.count_nonzero() > 0 else 0
            
    @property
    def topics(self):
        return self.src.topics & self.dst.topics
    
    @property
    def age(self):
        return self.dst.year - self.src.year
    
class Network():
    def __init__(self, nodes=[]):
        self.nodes = {}
        self.edges = set()
        self._construct(nodes)
        self._tfidf = {}
        self._node_nlp = {}
        
    def _construct(self, nodes):
        for node in nodes:
            self.nodes[node.id] = node
            for src in node.in_:
                if src in self:
                    self.edges |= {(src, node.id)} 
            for dst in node.out_:
                if dst in self:
                    self.edges |= {(node.id, dst)}
        
    def __getitem__(self, key):
        return self.nodes[key]
    
    def __iter__(self):
        for key in self.nodes:
            yield key

    def __contains__(self, key):
        return key in self.nodes
    
    def _fit_nlp(self, attr):
        self._tfidf[attr] = TfidfVectorizer(max_features=500)
        self._tfidf[attr].fit(getattr(node, attr) if isinstance(getattr(node, attr), str)
                              else ' '.join(getattr(node, attr))
                              for node in self.nodes.values())
        if attr in self._node_nlp:
            del self._node_nlp[attr]
        
    def _run_nlp(self, attr):
        for attr in self._tfidf:
            for node in  self.nodes.values():
                node_attr = getattr(node, attr)
                if attr not in self._node_nlp:
                    self._node_nlp[attr] = {}
                if node.id not in self._node_nlp[attr]:
                    self._node_nlp[attr][node.id] = self._tfidf[attr].transform([
                        node_attr if isinstance(node_attr, str)
                        else ' '.join(node_attr)])
       
    def compare(self, n, m, attr=None):
        if attr is None:
            return n == m
        n_attr, m_attr = (getattr(self[x], attr) for x in (n,m))
        if attr in self._tfidf:
            n_, m_ = self._node_nlp[attr][n], self._node_nlp[attr][m]
            similarity = (n_ @ m_.T)
            return similarity.data[0] if similarity.count_nonzero() > 0 else 0
        
        if isinstance(n_attr, set):
            return len(n_attr & m_attr)
        
        if isinstance(n_attr, (int, float)):
            return abs(n_attr - m_attr)
        
        return n_attr == m_attr
    
    def compare_many(self, n, m, attr):
        n_attr, m_attr = [np.array([getattr(self[x], attr) for x in u]) for u in (m,n)]
        if n_attr.dtype.kind in 'fi':
            return abs(n_attr - m_attr)
        else:
            return np.array([
             self.compare(x,y, attr) for (x,y) in zip(n,m)   
            ])
        
    def adj_matrix(self, idx=False):
        n_nodes = len(self.nodes)
        idx = dict(zip(self.nodes, range(n_nodes)))
        r,c,d = zip(*((idx[u], idx[v], 1) 
              for u,v in self.edges
              if u in idx and v in idx))
        M = coo_matrix((d, (r,c)), shape=(n_nodes, n_nodes), dtype=int)
        return M if not idx else (M, idx)
    
    def sim_matrix(self, attr):
        n_nodes = len(self.nodes)
        idx = dict(zip(self.nodes, range(n_nodes)))
        r,c,d = zip(*((idx[u], idx[v], self.compare(u,v, attr)) 
              for u,v in self.edges
              if u in idx and v in idx))
        M = coo_matrix((d, (r,c)), shape=(n_nodes, n_nodes), dtype=int)
        return M
        
    def add_nodes(self, nodes):
        self._construct(nodes)
    
    def remove_nodes(self, nodes):
        for node in nodes:
            del self.nodes[node]
        
    def page_rank(self):
        pg_ranks = Util.power_iteration(self.adj_matrix(), 50)
        return {node: rnk for node,rnk in zip(self.nodes.keys(), pg_ranks)}
    
    @property
    def shape(self):
        return f'(nodes x {len(self.nodes)}), (edges x {len(self.edges)})'
    
