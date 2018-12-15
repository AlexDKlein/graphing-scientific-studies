import pandas as pd
import numpy as np

class Util():
    @staticmethod
    def some(seq):
        for itm in seq:
            if itm: return itm
            
    @staticmethod
    def zip_value(seq, val):
        for itm in seq:
            yield (itm, val)
       
    @staticmethod
    def power_iteration(X, n):
        v = Util.unit_vect((X.shape[1],1))
        for _ in range(n):
            v_ = X @ v
            v = v_ / np.linalg.norm(v)
        return v

    @staticmethod
    def cosim(n, m, S, A, C=1e-5, k=100, nodelist=None):
        A/= np.sqrt((A * A).sum(axis=1))
        A = np.where(np.isfinite(A), A, 0)
        p_ = Util.p(n, A.shape[1])
        sim = 0
        for k_ in range(k):
            sim += C**k_ * p_[m] * p_[n]
            p_ = A @ p_
        return sim

    @staticmethod
    def p(i, n):
        return np.where(np.arange(n) == i, 1, 0)

    @staticmethod
    def unit_vect(shape):
        v = np.random.rand(*shape)
        v /= np.linalg.norm(v)
        return v

    @staticmethod
    def pagerank(X, eps=1e-3, p=0.85):
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        X_ = (p * X) + (1 - p) / X.shape[1]
        v, v_ = Util.unit_vect((X.shape[1], 1)), 1
        while np.linalg.norm(v - v_) > eps:
            v_ = v
            v = X_ @ v / np.linalg.norm(v)
        return v
    
    @staticmethod
    def cosimrank(S, A, C=1e-5):
        A,S = A.tocsr(), S.tocsr()
        A/= np.sqrt((A * A).sum(axis=1))
        A = np.where(np.isfinite(A), A, 0)
        return C*(A.T @ S @ A) + np.identity(A.shape[0])

    @staticmethod
    def expand_feature(seq):
        out = []
        for itm in seq:
            out += itm
        return out

    @staticmethod
    def repeat_itms(seq, n):
        if isinstance(seq, pd.Series):
            return Util.expand_feature(n * seq.apply(lambda x: [x]))
        return list(map(lambda x,y: [x] * y, seq, n))

    @staticmethod
    def repeat_rows(X, n):
        df = pd.DataFrame()
        for col in X.columns:
            df[col] = Util.repeat_itms(X[col], n)
        return df

    @staticmethod
    def create_authors(X, **kwargs):
        authors = pd.DataFrame(expand_feature(X['authors']))
        authors['papers'] = Util.expand_feature(Util.repeat_itms(X.index, X['authors'].apply(len)))
        authors['papers'] = authors['papers'].apply(lambda x: [x])
        X = repeat_rows(X, X['authors'].apply(len))
        
        authors['id'] = authors['ids'].apply(lambda x: x[0] if x else -1).astype(int)
        authors['missing_id'] = authors['ids'] == -1
        authors['coAuthors'] = (
            X['authors']
            .apply(lambda x: [itm['ids'] for itm in x])
            .apply(Util.expand_feature)
        )
        for field in kwargs:
            authors[field] = X[field]
        kwargs['name'] = 'first'
        kwargs['coAuthors'] = 'sum'
        kwargs['papers'] = 'sum'
        kwargs['missing_id']= 'first'
        authors = authors.groupby('id').agg(kwargs)
        return authors.reset_index()

