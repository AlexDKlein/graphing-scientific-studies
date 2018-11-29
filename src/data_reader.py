import json
from smart_open import smart_open
import time

class DataReader():
    def __init__(self, source, edges='data/edges', nodes='data/nodes'):
        self.source = source
        self.edges = edges
        self.nodes = nodes

    def read_edges(self, lim=None):
        with open(self.edges, 'r') as f:
            for i,line in enumerate(f): 
                if lim and i == lim: break
                yield line.strip().split('|')

    def read_nodes(self, lim=None):
        with open(self.nodes, 'r') as f:
            for i,line in enumerate(f):
                if lim and i == lim: break
                yield line.strip().split('|')

    def write(self, lim=None, dynamic=False):
        self.edges = self.create_edges(lim, dynamic)
        self.nodes = self.create_nodes(lim, dynamic)

    def create_edges(self, lim=None, dynamic=False):
        with open(self.edges, 'w') as f:
            with open(self.source) as s:
                for i,line in enumerate(s):
                    if lim and i == lim: break
                    entry = json.loads(line)
                    id_ = entry['id']
                    yr = entry['year'] if 'year' in entry else 1900
                    if not dynamic:
                        for dst in entry['outCitations']:
                            f.writelines(f'{id_}|{dst}\n')
                    for src in entry['inCitations']:
                        f.writelines(f'{src}|{id_}')
                        if dynamic:
                            f.writelines(f'|{yr}')
                        f.writelines('\n')
        return self.edges

    def create_nodes(self, lim=None, dynamic=False):
        with open(self.nodes, 'w') as f:
            with open(self.source) as s:
                for i,line in enumerate(s):
                    if lim and i == lim: break
                    entry = json.loads(line)
                    f.writelines('|'.join(str(entry[x]) if x in entry else '' for x in ('id','year','authors')))
                    f.writelines('\n')
        return self.nodes
 
class RetractionFinder():
    def __init__(self, pmids=None, dois=None, **kwargs):
        self.searched = set()
        self.found = []
        self.criteria = kwargs
        self.pmids = pmids
        self.dois = dois
        

    def search_files(self, *files):
        for doc in files:
            if doc in self.searched:
                continue
            with open(doc, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    pmid = entry['pmid'] if 'v' not in entry['pmid'] else entry['pmid'][:-2]
                    if 'retracted' in entry['title'][:10].lower() \
                        or self.pmids and pmid in self.pmids \
                        or self.dois and entry['doi'] in self.dois \
                        or any(entry[key] in values for key, values in self.criteria):
                        self.found.append(entry)
            self.searched |= {doc}

    def search_stream(self, stream):
        for line in stream.iter_lines():
            entry = json.loads(line)
            pmid = entry['pmid'] if 'v' not in entry['pmid'] else entry['pmid'][:-2]
            if 'retracted' in entry['title'][:10].lower() \
                or self.pmids and pmid in self.pmids \
                or self.dois and entry['doi'] in self.dois \
                or any(entry[key] in values for key, values in self.criteria):
                self.found.append(entry)
