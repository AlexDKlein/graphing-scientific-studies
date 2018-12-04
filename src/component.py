import json
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