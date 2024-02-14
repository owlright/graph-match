import networkx as nx

if __name__ == "__main__":
  import sys,os
  sys.path.append(os.getcwd())
  from inc.util.utils import get_attr_nodes
  from inc.topo import torus
  from inc.exps.exp_setup import *


def camdoop(G: nx.Graph, S, r, n) -> nx.DiGraph:
  '''Specially for finding aggregation tree in 3d-torus network'''
  get_x = lambda node: G.nodes[node]['pos'][0]
  get_y = lambda node: G.nodes[node]['pos'][1]
  get_z = lambda node: G.nodes[node]['pos'][2]
  connected_switch = {s: next(G.neighbors(s)) for s in S+[r]}
  node2xyz = {node: G.nodes[node]['pos'] for node in G if G.nodes[node]['type']=='switch'}
  xyz2node = {G.nodes[node]['pos']: node  for node in G if G.nodes[node]['type']=='switch'}
  r_switch = connected_switch[r]
  aggr_x, aggr_y, aggr_z = get_x(r_switch), get_y(r_switch), get_z(r_switch)

  mincost = G.number_of_edges()
  aggr_graph = None
  for delta_x in [1, -1]:
    for delta_y in [1, -1]:
      for delta_z in [1, -1]:
        g = nx.DiGraph()
        for s in S:
          node = connected_switch[s]
          x, y, z = node2xyz[node]
          while x != aggr_x:
            x = (x + delta_x) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
          while y != aggr_y:
            y = (y + delta_y) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
          while z != aggr_z:
            z = (z + delta_z) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
        if g.number_of_edges() < mincost:
          mincost = g.number_of_edges()
          aggr_graph = g
  for delta_x in [1, -1]:
    for delta_y in [1, -1]:
      for delta_z in [1, -1]:
        g = nx.DiGraph()
        for s in S:
          node = connected_switch[s]
          x, y, z = node2xyz[node]
          while x != aggr_x:
            x = (x + delta_x) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
          while z != aggr_z:
            z = (z + delta_z) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
          while y != aggr_y:
            y = (y + delta_y) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
        if g.number_of_edges() < mincost:
          mincost = g.number_of_edges()
          aggr_graph = g

  for delta_x in [1, -1]:
    for delta_y in [1, -1]:
      for delta_z in [1, -1]:
        g = nx.DiGraph()
        for s in S:
          node = connected_switch[s]
          x, y, z = node2xyz[node]
          while z != aggr_z:
            z = (z + delta_z) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
          while x != aggr_x:
            x = (x + delta_x) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
          while y != aggr_y:
            y = (y + delta_y) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
        if g.number_of_edges() < mincost:
          mincost = g.number_of_edges()
          aggr_graph = g

  for delta_x in [1, -1]:
    for delta_y in [1, -1]:
      for delta_z in [1, -1]:
        g = nx.DiGraph()
        for s in S:
          node = connected_switch[s]
          x, y, z = node2xyz[node]
          while y != aggr_y:
            y = (y + delta_y) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
          while x != aggr_x:
            x = (x + delta_x) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
          while z != aggr_z:
            z = (z + delta_z) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
        if g.number_of_edges() < mincost:
          mincost = g.number_of_edges()
          aggr_graph = g

  for delta_x in [1, -1]:
    for delta_y in [1, -1]:
      for delta_z in [1, -1]:
        g = nx.DiGraph()
        for s in S:
          node = connected_switch[s]
          x, y, z = node2xyz[node]
          while y != aggr_y:
            y = (y + delta_y) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
          while z != aggr_z:
            z = (z + delta_z) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
          while x != aggr_x:
            x = (x + delta_x) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
        if g.number_of_edges() < mincost:
          mincost = g.number_of_edges()
          aggr_graph = g
  for delta_x in [1, -1]:
    for delta_y in [1, -1]:
      for delta_z in [1, -1]:
        g = nx.DiGraph()
        for s in S:
          node = connected_switch[s]
          x, y, z = node2xyz[node]
          while z != aggr_z:
            z = (z + delta_z) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
          while y != aggr_y:
            y = (y + delta_y) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
          while x != aggr_x:
            x = (x + delta_x) % n
            next_node = xyz2node[(x, y, z)]
            g.add_edge(node, next_node)
            node = next_node
        if g.number_of_edges() < mincost:
          mincost = g.number_of_edges()
          aggr_graph = g
  for s in S:
    node = connected_switch[s]
    aggr_graph.add_edge(s, node)
  node = connected_switch[r]
  aggr_graph.add_edge(node, r)
  assert nx.is_tree(aggr_graph) and max(d for n, d in aggr_graph.out_degree()) <= 1

  aggr_graph._node.update((n, d.copy()) for n, d in G.nodes.items() if n in aggr_graph)
  return aggr_graph

if __name__=="__main__":
  G = torus(8)
  G, _ = get_reindexed_graph(G)
  hosts = get_attr_nodes(G, 'type', 'host')
  random.seed(23234)
  S = random.sample(hosts, 11)

  r = S.pop()
  odpath, oddist = get_paths(G)

  g = camdoop(G, S, r, 8)
  print(g.number_of_edges())

