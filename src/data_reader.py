import json
class DataReader():
    def __init__(self, source, edges='data/edges', nodes='data/nodes'):
        self.source = source
        self.edges = edges
        self.nodes = nodes

    def read_edges(self, lim=None):
        with open(self.edges, 'r') as f:
            for i,line in enumerate(f): 
                if lim and i == lim: break
                yield line.strip().split(',')

    def read_nodes(self, lim=None):
        with open(self.nodes, 'r') as f:
            for i,line in enumerate(f):
                if lim and i == lim: break
                yield line.strip().split('|')

    def write(self):
        self.edges = self.create_edges()
        self.nodes = self.create_nodes()

    def create_edges(self):
        with open(self.edges, 'w') as f:
            with open(self.source) as s:
                for line in s:
                    entry = json.loads(line)
                    id_ = entry['id']
                    for dst in entry['outCitations']:
                        f.writelines(f'{id_},{dst}\n')
                    for src in entry['inCitations']:
                        f.writelines(f'{src},{id_}\n')
        return self.edges

    def create_nodes(self):
        with open(self.nodes, 'w') as f:
            with open(self.source) as s:
                for line in s:
                    entry = json.loads(line)
                    f.writelines('|'.join(str(entry[x]) if x in entry else '' for x in ('id','year','authors','entities')))
                    f.writelines('\n')
        return self.nodes
