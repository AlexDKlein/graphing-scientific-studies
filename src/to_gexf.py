def to_gexf(g, path, dynamic=False):
    """
    Export a pyspark graphframe to a .gexf file for use in Gephi.
    ============
    Takes:
        g: pyspark graphframe object-graph to export
        path: string-location to save file
    ============
    Returns: None
    """
    if dynamic:
      nodes = ''.join(g.vertices.rdd.map(lambda v: f'      <node id="{v["id"]}"  start="{v["year"]}" end="2018" />\n').collect())
      edges = ''.join(g.edges.rdd.map(lambda e: f'      <edge id="{e["id"]}" source="{e["src"]}" target="{e["dst"]}" start="{e["year"]}" end="2018" />\n').collect())
    else:
      nodes = ''.join(g.vertices.rdd.map(lambda v: f'      <node id="{v["id"]}" />\n').collect())
      edges = ''.join(g.edges.rdd.map(lambda e: f'      <edge id="{e["id"]}" source="{e["src"]}" target="{e["dst"]}" />\n').collect())
    string =f"""<?xml version="1.0" encoding="UTF-8"?>
    <gexf xmlns="http://www.gexf.net/1.2draft" version="1.2" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.w3.org/2001/XMLSchema-instance">
      <graph mode="{'static' if not dynamic else 'dynamic'}" defaultedgetype="directed" name="">
        <nodes>
  {nodes}
        </nodes>
        <edges>
  {edges}
        </edges>
      </graph>
    </gexf>"""
    with open(path, 'w') as f:
        f.write(string)

def pd_to_gexf(edges, nodes, path, node_attr={}, edge_attr={}, dynamic=True, node_label=None):
    """
    Export two pandas frames to a single .gexf file for use in Gephi.
    ============
    Takes:
        edges: DataFrame object containing edges to export
        nodes: DataFrame object containing nodes to export
        path: string-location to save file
        attributes: keyword parameters corresponding to attribute names
    ============
    Returns: None
    """
    _nodes, _edges = '', ''
    _node_attr, _edge_attr = '', ''
    for i, (attribute, type_) in enumerate(node_attr.items()):
        _node_attr += f'      <attribute id="{i}" title="{attribute}" type="{type_}"/>\n'
    for i, (attribute, type_) in enumerate(edge_attr.items()):
        _edge_attr += f'      <attribute id="{i}" title="{attribute}" type="{type_}"/>\n'
    for _, n in nodes.iterrows():
        _nodes += f'      <node id="{n["id"]}" label="{n[node_label] if node_label else ""}" start="{n["year"]}" end="2018">\n'
        _nodes += f'        <attvalues>\n'
        for i, attr in enumerate(node_attr):
            _nodes += f'          <attvalue for="{i}" value="{n[attr]}"/>\n'
        _nodes += f'        </attvalues>\n'
        _nodes += '      </node>\n'
    for _, e in edges.iterrows():
        _edges += f'      <edge id="{e["id"]}" source="{e["src"]}" target="{e["dst"]}" start="{e["year"] if "year" in e else "2018"}" end="2018" weight="{e["weight"] if "weight" in e else "1"}">\n'
        _edges += f'        <attvalues>\n'
        for i, attr in enumerate(edge_attr):
            _edges += f'          <attvalue for="{i}" value="{e[attr]}"/>\n'
        _edges += f'        </attvalues>\n'
        _edges += '      </edge>\n'
    
    string =f"""<?xml version="1.0" encoding="UTF-8"?>
    <gexf xmlns="http://www.gexf.net/1.2draft" version="1.2" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.w3.org/2001/XMLSchema-instance">
      <graph mode="{'static' if not dynamic else 'dynamic'}" defaultedgetype="directed" name="">
        <attributes class="node">
  {_node_attr}
        </attributes>        
        <nodes>
  {_nodes}
        </nodes>
        <attributes class="edge">
  {_edge_attr}
        </attributes>     
        <edges>
  {_edges}
        </edges>
      </graph>
    </gexf>"""
    with open(path, 'w') as f:
        f.write(string)