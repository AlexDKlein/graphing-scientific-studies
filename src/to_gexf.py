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
    nodes = ''.join(g.vertices.rdd.map(lambda v: f'      <node id="{v["id"]}" />\n').collect())
    edges = ''.join(g.edges.rdd.map(lambda e: f'      <edge id="{e["id"]}" source="{e["src"]}" target="{e["dst"]}" {'start="{e['year']}" end="2018"' if dynamic else ''}/>\n').collect())
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