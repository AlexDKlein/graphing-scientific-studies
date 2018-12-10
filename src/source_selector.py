import pandas as pd 
import numpy as np 
from .network import PaperNLP, EdgeSplitter
from sklearn.metrics.pairwise import cosine_similarity
import requests

class Recommender():
    def __init__(self, model=None, *fields):
        self.nlp = PaperNLP(*fields)
        self._model = model
        self._nodes = None

    def fit(self, X, y=None):
        self._nodes = X.copy()
        splitter = EdgeSplitter()
        E = splitter.transform(X)
        E,y = E.drop('edge', axis=1), E['edge']
        self.nlp.fit(X)
        E = self.transform(E, X)
        self._model.fit(E, y.astype(int).values)
        self._transformed_nodes = self.nlp.transform(self._nodes)
        return self

    def _predict(self, x):
        X = self._nodes.append(x, sort=True)
        S = self.nlp.similarity_matrix(X)
        E = pd.DataFrame()
        E['src'] = X['_id']
        E['dst'] = [x['_id'].values[0] for _ in range(len(E))]
        for field in self.nlp._tfidf:
            E[field + '_similarity'] = S[field].lookup(col_labels=E['src'], row_labels=E['dst'])
        return self._model.predict_proba(E.drop(columns=['src','dst'], axis=1))

    def predict(self, x):
        t = self.nlp.transform(x)
        T = self._transformed_nodes
        E = pd.DataFrame()
        for field in T:
            E[field + '_similarity'] = Recommender.cos_sim(T[field], t[field]).values.ravel()
        E['age'] = abs(x['year'] if 'year' in x else 0 - 
                       self._nodes['year'].loc[self._nodes['_id'].values].values)
        E['age'] = E['age'] = np.where(E['age'].isnull(), 0, E['age'])
        return self._model.predict_proba(E)[:, 1]
    
    def recommend(self, x, k=5):
        pred = self.predict(x)
        return self._nodes.iloc[np.argsort(pred)[::-1][1:k+1]]

    def predict_one(self, x, y):
        if isinstance(x, pd.Series): x = pd.DataFrame(x).T
        if isinstance(y, pd.Series): y = pd.DataFrame(y).T
        xt,yt = self.nlp.transform(x), self.nlp.transform(y)
        E = pd.DataFrame()
        for field in self.nlp._tfidf:
            E[field + '_similarity'] = Recommender.cos_sim(xt[field], yt[field]).values.ravel()
        E['age'] = abs(x['year'].values - y['year'].values)
        return self._model.predict_proba(E)[:, 1][0]

    def transform(self, E, X):
        E = E.copy()
        S = self.nlp.similarity_matrix(X)
        for field in self.nlp._tfidf:
            E[field + '_similarity'] = S[field].lookup(col_labels=E['src'], row_labels=E['dst'])
        E['age'] = abs(X['year'].loc[E['dst'].values].values - 
                       X['year'].loc[E['src'].values].values)
        E['age'] = E['age'] = np.where(E['age'].isnull(), 0, E['age'])
        E.drop(columns=[col for col in ('src','dst','edge') if col in E.columns],
               axis=1, inplace=True)
        return E

    def prompt(self):
        pass

    def _transform_edges(self, E, X):
        E = E.copy()
        src = self.nlp.transform(X.loc[E['src']])
        dst = self.nlp.transform(X.loc[E['dst']])
        for field in self.nlp._tfidf:
            E[field + '_similarity'] = ((src[field] @ dst[field].T).sum(axis=1)).values.ravel()
        E['age'] = abs(X.loc[E['src']]['year'].values - X.loc[E['dst']]['year'].values)
        E['age'] = E['age'] = np.where(E['age'].isnull(), 0, E['age'])
        return E

    def predict_proba(self, E, X):
        E = self.transform(E,X)
        return self._model.predict_proba(E)[:, 1]

    @staticmethod
    def norm(x):
        return np.linalg.norm(x, axis=-1)
    
    @staticmethod
    def cos_sim(x,y):
        return (x @ y.T)

class InfluenceModel():
    def __init__(self, base):
        self.base = base
    
    def fit(self, X, y):
        self.base.fit(X, y)

    def predict(self, X):
        pass

    def transform(self, X, y):
        pass

    @staticmethod
    def generate_edge_ids(seed_id):
        resp=requests.get(f'http://api.semanticscholar.org/v1/paper/{seed_id}')
        references=pd.DataFrame(resp.json()['references'])
        X=references.loc[:, ['paperId']]
        y=references['isInfluential']
        return X,y

