import networkx as nx
from .construct import *
def build_sptree(S, r, odpath) -> nx.DiGraph:
  '''return tree composed of shortest paths'''
  P = {}
  for s in S:
    P[s] = r
  return construct_tree_from_paths(P, odpath)
