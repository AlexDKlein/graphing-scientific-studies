import numpy as np 
import pandas as pd 
import networkx as nx
import requests
import json
# from to_gexf import pd_to_gexf

def create_edges(X, in_='inCitations', out_='outCitations', directed=False):
    edge_list = []
    for dst, srcs in X[in_].iteritems():
        for src in srcs: 
            if not directed:
                edge_list.append(sorted((src, str(dst))))
            else:
                edge_list.append((src, dst))
    if out_ is not None:
        for src, dsts in X[out_].iteritems():
            for dst in dsts: edge_list.append([src, dst])
    edges = pd.DataFrame(edge_list, columns=['src', 'dst'])
    edges['id'] = edges.index
    edges['weight'] = 1
    edges = edges.groupby(['src','dst']).agg(
            {'id':'first',
             'weight': 'sum'}).reset_index()
    return edges

def expand_feature(seq):
    out = []
    for itm in seq:
        out += itm
    return out

def repeat_itms(seq, n):
    if isinstance(seq, pd.Series):
        return expand_feature(n * seq.apply(lambda x: [x]))
    return list(map(lambda x,y: [x] * y, seq, n))

def repeat_rows(X, n):
    df = pd.DataFrame()
    for col in X.columns:
        df[col] = repeat_itms(X[col], n)
    return df

def create_authors(X, **kwargs):
    authors = pd.DataFrame(expand_feature(X['authors']))
    authors['papers'] = expand_feature(repeat_itms(X.index, X['authors'].apply(len)))
    authors['papers'] = authors['papers'].apply(lambda x: [x])
    X = repeat_rows(X, X['authors'].apply(len))
    
    authors['id'] = authors['ids'].apply(lambda x: x[0] if x else -1).astype(int)
    authors['missing_id'] = authors['ids'] == -1
    authors['coAuthors'] = (
        X['authors']
        .apply(lambda x: [itm['ids'] for itm in x])
        .apply(expand_feature)
    )
    for field in kwargs:
        authors[field] = X[field]
    kwargs['name'] = 'first'
    kwargs['coAuthors'] = 'sum'
    kwargs['papers'] = 'sum'
    kwargs['missing_id']= 'first'
    authors = authors.groupby('id').agg(kwargs)
    return authors.reset_index()



class GraphSpace():
    def __init__(self, nodes, edges, collection):
        self.nodes = nodes
        self.edges = edges
        self.collection = collection

    def dist(self, source, target):
        pass

    def dir_dist(self, source, target):
        pass

    def export(self, path):
        pass

    @staticmethod
    def construct_component(collection, start_id=None, 
                        max_depth=3, max_size=1e6,
                        component=set(), depth=0, v=False):
        if start_id in component or depth > max_depth and max_depth > 0:
            return component
        component |= {start_id}
        node = collection.find_one(start_id)
        if node is None: 
            return component
        for other in node['inCitations'] + node['outCitations']:
            component = GraphSpace.construct_component(collection, other, 
                                            max_depth, component, 
                                            depth=depth + 1, v=v)
        if v and len(component):
            print(len(component), depth)
        return component

    @staticmethod
    def write_component(collection, component, path='data/component'):
        result = collection.find({'_id': {'$in': list(component)}})
        with open(path, 'w') as f:
            for itm in result:
                f.writelines(json.dumps(itm))
                f.writelines('\n')
    
    def collect_papers(self, papers):
        return pd.DataFrame([itm for itm in self.collection.find({'_id': {'$in': list(papers)}})])

    @staticmethod
    def expand_feature(seq):
        out = []
        for itm in seq:
            out += itm
        return out



class AuthorNetwork(GraphSpace):
    def __init__(self, seed_id, collection, approx_size=10000, storage_path='data/authors_comp', **kwargs):
        self.collection = collection
        self.component = self._construct(seed_id, lim=approx_size, **kwargs)
        self.write_component(self.collection, self.component, storage_path)
        self.paper_nodes = pd.read_json(storage_path, lines=-1)
        self.paper_nodes.index = self.paper_nodes['_id']
        self.paper_edges = create_edges(self.paper_nodes)
        self.paper_edges['weight'] = 1
        self.author_nodes = create_authors(self.paper_nodes)
        self.author_edges = create_edges(self.author_nodes, in_='coAuthors', out_=None)

    def _construct(self, seed_id, max_paper_distance=1, max_coauther_distance=1, v=False, component=set(), lim=10000):
        if len(component) > lim:
            return component
        papers = self.author_lookup(seed_id)
        if max_paper_distance > 0:
            for paper in papers:
                component = self.construct_component(self.collection, start_id=paper, 
                                        max_depth=max_paper_distance, 
                                        component=component, depth=0, v=v)
        if max_coauther_distance > 0:
            df = self.collect_papers(component)
            authors = create_authors(df)
            for author in authors['id']:
                component = self._construct(author, max_paper_distance=max_coauther_distance,
                                max_coauther_distance=max_coauther_distance - 1, v=v, 
                                component=component)
        return component
    
    def export(self, path, space ='papers'):
        if space == 'papers':
            pd_to_gexf(edges=self.paper_edges, nodes=self.paper_nodes, dynamic=True, path=path)
        elif space == 'authors':
            pd_to_gexf(edges=self.author_edges, nodes=self.author_nodes, dynamic=True, path=path)
        return path

    @staticmethod
    def author_lookup(_id):
        try:
            resp = requests.get(f'http://api.semanticscholar.org/v1/author/{_id}')
            return set(itm['paperId'] for itm in resp.json()['papers'])
        except KeyError:
            return set()
   
   