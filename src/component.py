import json
import requests
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

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
        component = construct_component(collection, other, 
                                        max_depth, component, 
                                        depth=depth + 1, v=v)
    if v and len(component):
        print(len(component), depth)
    return component

def write_component(collection, component, path='data/component'):
    result = collection.find({'_id': {'$in': list(component)}})
    with open(path, 'w') as f:
        for itm in result:
            f.writelines(json.dumps(itm))
            f.writelines('\n')

class ComponentCreator():
    def __init__(self, collection):
        self.collection = collection

    def mixed_construct(self, seed_id, max_paper_distance=1, max_coauther_distance=1, v=False, component=set(), lim=10000):
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
                component = self.mixed_construct(author, max_paper_distance=max_coauther_distance,
                                max_coauther_distance=max_coauther_distance - 1, v=v, 
                                component=component)
        return component

    def construct_component(self, start_id=None, 
                        max_depth=3, max_size=1e6,
                        component=set(), depth=0, v=False):
        if start_id in component or depth > max_depth and max_depth > 0:
            return component
        component |= {start_id}
        node = self.collection.find_one(start_id)
        if node is None: 
            return component
        for other in node['inCitations'] + node['outCitations']:
            component = self.construct_component(other, 
                                            max_depth, component, 
                                            depth=depth + 1, v=v)
        if v and len(component):
            print(len(component), depth)
        return component
    
    def edge_construct(self, _ids, depth=1):
        used = set()
        E,X = [pd.DataFrame() for _ in range(2)]
        y = pd.Series()
        E['src'] = []
        E['dst'] = []
        for _ in range(depth):
            for _id in _ids:
                if _id not in used:
                    used.add(_id)
                    E_,X_,y_ = self.edges_lookup(_id)
                    if X_ is None or '_id' not in X_.columns: 
                        continue
                    E = E.append(E_, sort=True)
                    X,y = X.append(X_), y.append(y_)
            _ids = E['src'].values
        X.drop_duplicates(subset='_id', inplace=True)
        # E,X,y =[itm.reset_index(drop=True) for itm in (E,X,y)]
        return E,X,y

    
    @staticmethod
    def author_lookup(_id):
        try:
            resp = requests.get(f'http://api.semanticscholar.org/v1/author/{_id}')
            return set(itm['paperId'] for itm in resp.json()['papers'])
        except KeyError:
            return set()

    def paper_lookup(self, *_ids):
        return pd.DataFrame([itm for itm in self.collection.find({'_id': {'$in': list(_ids)}})])

    def edges_lookup(self, seed_id):
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