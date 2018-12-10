import json
import numpy as np 
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier  

class Network():
    def __init__(self, X, author_path=None, paper_path=None, collection=None):
        self.papers = X.copy()
        self.authors = self.create_authors_dataframe(X, collection)
        self.paper_path = paper_path
        self.author_path = author_path
        self.write_json(X, paper_path)
        self.write_json(X, author_path)
        
    def write(self):
        self.create_edges(self.author_path, self.author_path + '_edges', 
                        'inAuthors', 'outAuthors', None, True)
        self.create_edges(self.paper_path, self.paper_path + '_edges', 
                        'inCitations', 'outCitations', None, True)
        self.create_nodes(self.author_path, self.author_path + '_nodes')
        self.create_nodes(self.paper_path, self.paper_path + '_nodes')
    
    @staticmethod
    def create_edges(source, path, in_field, out_field, lim=None, dynamic=False, **fields):
        with open(path, 'w') as f:
            with open(source) as s:
                for i,line in enumerate(s):
                    if lim and i == lim: break
                    entry = json.loads(line)
                    id_ = entry['id'] if 'id' in entry else entry['_id']
                    yr = entry['year'] if 'year' in entry else 1900
                    if not dynamic:
                        for dst in entry[out_field]:
                            f.writelines(f'{id_}|{dst}\n')
                    for src in entry[in_field]:
                        f.writelines(f'{src}|{id_}')
                        if dynamic:
                            f.writelines(f'|{yr}')
                        f.writelines('\n')
        return path

    @staticmethod
    def create_nodes(source, path, lim=None, **fields):
        with open(path, 'w') as f:
            with open(source) as s:
                for i,line in enumerate(s):
                    if lim and i == lim: break
                    entry = json.loads(line)
                    f.writelines('|'.join(str(entry[x]) if x in entry else ''))
                    f.writelines('\n')
        return path

    @staticmethod
    def write_json(X, features=None, path=None):
        return (X.loc[:, features].to_json(path) 
                if features else X.to_json(path))

    @staticmethod
    def expand_feature(seq, sep=None):
        out = []
        for itm in seq:
            out.extend(itm)
        df = pd.DataFrame(out)
        return df
        
    @staticmethod
    def create_authors_dataframe(X, collection=None):
        '''Expand "authors" columns of a dataframe and return dataframe containing author stats.
        
        Parameters
        ----------
        X: DataFrame containing "author" column.
        
        Returns
        ----------
        authors: pd.DataFrame
        
        '''
        X = X.copy()
        authors = Network.expand_feature(X['authors'])
        authors['ids'] = authors['ids'].apply(lambda x: x[0] if x else -1).astype(int)
        X['numAuthors'] = X['authors'].apply(len)
        authors['paper'] = Network.expand_feature(
            X.loc[:, ['numAuthors','_id']].T
             .apply(lambda x: [x.iloc[1]]*x.iloc[0]))
        authors['paper'] = authors['paper'].apply(lambda x: [x])
        authors['missing_id'] = authors['ids'] == -1
        authors['coAuthors'] = Network.expand_feature(
            X.loc[:, ['numAuthors','authors']].T
             .apply(lambda x: [[[y['ids'] for y in x.iloc[1]]]]*x.iloc[0])
            )
        authors['coAuthors'] = authors['coAuthors'].apply(lambda x: [itm[0] for itm in x if itm])
        if collection is not None:
            X['inAuthors'] = (X['inCitations']
                .apply(lambda x: collection.find({'_id': {'$in': x}}))
                .apply(lambda x: [z['ids'][0] for y in x for z in y['authors'] if z['ids']])
            )
            X['outAuthors'] = (X['outCitations']
                .apply(lambda x: collection.find({'_id': {'$in': x}}))
                .apply(lambda x: [z['ids'][0] for y in x for z in y['authors'] if z['ids']])
            )
            authors['inAuthors'] = Network.expand_feature(
                X.loc[:, ['numAuthors', 'inAuthors']].T
                .apply(lambda x: [[x.iloc[1]]]*x.iloc[0])
            )
            authors['outAuthors'] = Network.expand_feature(
                X.loc[:, ['numAuthors', 'outAuthors']].T
                .apply(lambda x: [[x.iloc[1]]]*x.iloc[0])
            )
            print('Done')
        
        authors['ids'].loc[authors['missing_id']] = np.cumsum(authors[authors['missing_id']]['ids'])
        papers, coauthors = authors.copy(), authors.copy()
        papers = papers.groupby('ids')['paper'].sum()
        coauthors = coauthors.groupby('ids')['coAuthors'].sum()
        authors = authors.groupby('ids').first().reset_index()
        authors['papers'] = papers.values
        authors['numPapers'] = authors['papers'].apply(len)
        authors['coAuthors'] = coauthors.values
        authors['numCoAuthors'] = coauthors.apply(lambda x: sum(1 for itm in x if itm)).values
        authors.columns = ['_id'] + list(authors.columns[1:])
        return authors

class PaperNLP():
    def __init__(self, *fields):
        self._tfidf = {}
        for field in fields:
            self._tfidf[field] = TfidfVectorizer()
      
    def fit(self, X, y=None):
        X = X.copy()
        self.nodes = X
        for field, tfidf in self._tfidf.items():
            tfidf.fit(
                X[field].apply(lambda x: 
                    (x if isinstance(x, str)
                     else ' '.join(x) 
                     if not isinstance(x, float)
                     else '')
                    .lower()
                    .replace('withdraw', '')
                    .replace('retract', ''))
                    )
        return self

    def transform(self, X, y=None):
        output = {}
        for field, tfidf in self._tfidf.items():
            output[field] = tfidf.transform(
                X[field].apply(lambda x: 
                    (x if isinstance(x, str)
                     else ' '.join(x)
                     if not isinstance(x, float)
                     else '')
                    .lower()
                    .replace('withdraw', '')
                    .replace('retract', ''))
                    )
        # return ({k: v for k,v in output.items()}, X['_id'].values)
        return {k: pd.DataFrame(v.toarray(), index=X['_id']) for k,v in output.items()}

    def similarity_matrix(self, X):
        transformed = self.transform(X)
        similarity = {}
        for k,v in transformed.items():
            similarity[k] = pd.DataFrame(cosine_similarity(v), 
                                columns=X['_id'], 
                                index=X['_id'])
        return similarity
        
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

def adj_matrix(edges, nodes=None, source_col='source', target_col='target'):
    '''Create a pandas DataFrame containing the adjacency matrix described by a 
    group of edges.
    '''
    edges = edges.copy()
    if nodes is None:
        nodes = pd.DataFrame()
        nodes['_id'] = np.unique(edges.loc[:, [source_col,target_col]].values)
    
    edges.columns = [(c 
        if c!=source_col else 'source')
        if c!=target_col else 'target'
        for c in edges.columns]
    nodes_set = set(nodes['_id'].values)
    mask = np.array([itm in nodes_set for itm in edges['source']]) * \
           np.array([itm in nodes_set for itm in edges['target']])
    g = nx.Graph(edges[mask].loc[:, ['source', 'target']])
    adj = nx.adj_matrix(g, nodelist=nodes['_id'].values)
    return adjm
    adj_df = pd.DataFrame(adj.toarray())
    adj_df.index = nodes['_id'].values
    adj_df.columns = nodes['_id'].values
    return adj_df

if __name__ == '__main__':
    fields = ['entities', 'paperAbstract', 'title']
    splitter = EdgeSplitter()
    nlp = PaperNLP(*fields)
    X0,E0,Xt,Et = splitter.split(component, edges_df)
    nlp.fit(X0)
    S0,St = [nlp.similarity_matrix(X) for X in (X0,Xt)]
    y0,yt = E0['edge'], Et['edge']
    for field in fields:
        E0[field + '_similarity'] = S0[field].lookup(col_labels=E0['src'], row_labels=E0['dst'])
        Et[field + '_similarity'] = St[field].lookup(col_labels=Et['src'], row_labels=Et['dst'])

    gbc = GradientBoostingClassifier(n_estimators=1000, random_state=5476, max_features=2)
    gbc.fit(E0.loc[:, [f + '_similarity' for f in fields]], y0.astype(int))

    tpr, fpr, thr = roc_curve(y_score=gbc.predict_proba(
        Et.loc[:, [f + '_similarity' for f in fields]])[:,1], 
                            y_true=yt.astype(int))
    print(roc_auc_score(y_score=gbc.predict_proba(
        Et.loc[:, [f + '_similarity' for f in fields]])[:,1],
        y_true=yt.astype(int)))
    plt.plot(tpr,fpr)
