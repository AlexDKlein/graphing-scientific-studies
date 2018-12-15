import json
import requests
import pandas as pd
import numpy as np
import random
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from .util import Util
from .splitter import Splitter


class ComponentCreator():
    def __init__(self, collection):
        self.collection = collection

    def _mixed_construct(self, seed_id, max_paper_distance=1, max_coauther_distance=1, v=False, component=set(), lim=1000):
        if len(component) > lim:
            return component
        papers = self.author_lookup(seed_id)
        if max_paper_distance > 0:
            for paper in papers:
                component = self.construct_component(start_id=paper, 
                                        max_depth=max_paper_distance, 
                                        component=component, depth=0, v=v)
        if max_coauther_distance > 0:
            df = self.paper_lookup(*component)
            authors = Util.create_authors(df)
            for author in authors['id']:
                component = self._mixed_construct(author, max_paper_distance=max_coauther_distance,
                                max_coauther_distance=max_coauther_distance - 1, v=v, 
                                component=component, lim=lim)
        return component

    def mixed_construct(self, seed_id=None, create_missing=False, **kwargs):
        '''Broken. Subsequent calls return identical outputs until the kernal resets.'''
        if seed_id is None:
            try: 
                seed_id=self.random_record()['authors'][0]['ids'][0]
            except (IndexError, KeyError):
                return self.mixed_construct(seed_id=None, create_missing=create_missing, **kwargs)
        X = self.paper_lookup(
            *self._mixed_construct(seed_id, **kwargs)
        )
        X.index = X['_id']
        E = Splitter.create_edge_dataframe(X)
        if not create_missing: E = E[E['weight'] > 1]
        return E,X
    
    def _random_construct(self, min_size=1e4):
        component=set()
        while len(component) < min_size:
            start_id = self.random_record()['_id']
            component = self.construct_component(start_id, max_depth=1,
                                                 lim=1e4, component=component)
        return component

    def random_construct(self, min_size=1e4, create_missing=False):
        X = self.paper_lookup(
            *self._random_construct(min_size)
        )
        X.index = X['_id']
        E = Splitter.create_edge_dataframe(X)
        if not create_missing: E = E[E['weight'] > 1]
        return E,X

    def construct_component(self, start_id=None, 
                        max_depth=3, lim=1e4,
                        component=set(), depth=0, v=False):
        if start_id in component or depth > max_depth and max_depth > 0:
            return component
        component |= {start_id}
        node = self.collection.find_one(start_id)
        if node is None: 
            return component
        for other in node['inCitations'] + node['outCitations']:
            component = self.construct_component(start_id=other, max_depth=max_depth, 
                                                component=component, lim=lim, 
                                                depth=depth + 1, v=v)
        if v and len(component):
            print(len(component), depth)
        return component
    
    def edge_construct(self, _ids=2, depth=3, get_influence=False, lim=1000):
        if isinstance(_ids, int):
            _ids = [self.random_record()['_id'] for _ in range(_ids)]
        used = set()
        E,X = [pd.DataFrame() for _ in range(2)]
        y = pd.Series()
        E['src'] = []
        E['dst'] = []
        for _ in range(depth):
            if len(X) > lim: break
            for _id in _ids:
                if _id not in used:
                    used.add(_id)
                    if get_influence:
                        E_,X_,y_ = self.edges_influence_lookup(_id)
                    else:
                        E_,X_ = self.edges_lookup(_id)
                    if X_ is None or '_id' not in X_.columns: 
                        continue
                    E,X = E.append(E_, sort=True), X.append(X_, sort=True)
                    y = y.append(y_) if get_influence else None
            _ids = E['src'].values
        X.drop_duplicates(subset='_id', inplace=True)
        E,X =[itm.reset_index(drop=True) for itm in (E,X)]
        if y is not None: y = y.reset_index(drop=True)
        X.index = X['_id']
        return (E,X,y) if y is not None else (E,X)

    def coauthor_construct(self, _ids, depth=1):
        used = set()
        E,X = [pd.DataFrame() for _ in range(2)]
        y = pd.Series()
        E['src'] = []
        E['dst'] = []
        for _ in range(depth):
            for _id in _ids:
                if _id not in used:
                    used.add(_id)
                    P_, A_ = self.coauthor_lookup(_id)
                    if P_ is None or '_id' not in P_.columns: 
                        continue
                    E_,X_ = self.edge_construct([i for i in P_['_id']])
                    E = E.append(E_, sort=True)
                    X = X.append(X_)
        X.drop_duplicates(subset='_id', inplace=True)
        E,X =[itm.reset_index(drop=True) for itm in (E,X)]
        X.index = X['_id']
        return E,X

    @staticmethod
    def author_lookup(_id):
        try:
            resp = requests.get(f'http://api.semanticscholar.org/v1/author/{_id}')
            return set(itm['paperId'] for itm in resp.json()['papers'])
        except KeyError:
            return set()

    def paper_lookup(self, *_ids):
        return pd.DataFrame([itm for itm in self.collection.find({'_id': {'$in': list(_ids)}})])

    def edges_influence_lookup(self, seed_id):
        resp=requests.get(f'http://api.semanticscholar.org/v1/paper/{seed_id}')
        try:
            references=pd.DataFrame(resp.json()['references'])
            E=pd.DataFrame([itm[0] for itm in references.loc[:, ['paperId']].values], columns=['src'])
            E['dst']=seed_id
            X=self.paper_lookup(*(itm[0] for itm in references.loc[:, ['paperId']].values))
            y=references['isInfluential'].astype(int)
            return E,X,y
        except KeyError:
            return None,None,None

    def edges_lookup(self, seed_id):
            resp=self.paper_lookup(seed_id)
            try:
                references=resp['inCitations']
                E=pd.DataFrame(*references.values, columns=['src'])
                E['dst']=seed_id
                X=self.paper_lookup(*references.values[0])
                references=resp['outCitations']
                E_ = pd.DataFrame(*references.values, columns=['dst'])
                E_['src']=seed_id
                X_=self.paper_lookup(*references.values[0])
                # y=references['isInfluential'].astype(int)
                E = E.append(E_,sort=True)
                X = X.append(X_, sort=True)
                return E,X#,y
            except KeyError as e:
                print(e)
                return None,None,None

    def coauthor_lookup(self, seed_id):
        resp=requests.get(f'http://api.semanticscholar.org/v1/author/{seed_id}')
        try:
            papers=self.paper_lookup(*(itm['paperId'] for itm in resp.json()['papers']))
            authors = set()
            papers['authors'].apply(lambda x: [authors.add(itm['ids'][0]) for itm in x if x])
            return papers, authors
        except KeyError:
            return None,None

    def random_record(self):
        idx = random.choice('abcdef') + str(random.randint(0, 1e6))
        return self.collection.find_one(filter={'_id':{'$gt': idx}})

    @staticmethod
    def write_component(collection, component, path='data/component'):
        result = collection.find({'_id': {'$in': list(component)}})
        with open(path, 'w') as f:
            for itm in result:
                f.writelines(json.dumps(itm))
                f.writelines('\n')