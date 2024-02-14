import networkx as nx
import itertools
def construct_tree_from_paths(P, odpath=None) -> nx.DiGraph:
  '''
  P : the dict of (src, dst), P[s][t][0] is dest node, P[s][t][1] is distance
  stpath : make sure every pair in P has its path in stpath
              or, pairpath is None just paths in P
  '''
  T = nx.DiGraph()
  T.graph['rankdir'] = 'BT'
  if not odpath:
    for _, p in P.items():
      T.add_edges_from(itertools.pairwise(p))
  else:
    for s, t in P.items():
      T.add_edges_from(list(itertools.pairwise(odpath[s, t])))
  return T
