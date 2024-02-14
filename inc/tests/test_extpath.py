#import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numba.typed import List,Dict

import sys,os
sys.path.append(os.getcwd())
from inc.util import *
from inc.algorithms.bruteforce_exp import *
from inc.topo import random_graph
from inc.algorithms.extract_paths import extract_paths
#from inc.util.plotters import plot, draw_flow_paths

def setup(S, r, odpath):
  forbiddens = set()
  last_cost = 0
  # ! remove nodes cannot be computation nodes
  for s in S:
    forbiddens.add(s)
  forbiddens.add(r)
  for node in capacity:
    if capacity[node] == 0:
      forbiddens.add(node)
  # ! get potential nodes (which is W in my paper) for each tree
  paggrs = set()
  for s in S:
    for n in odpath[s, r]:
      if n not in forbiddens:
        paggrs.add(n)
  return paggrs


if __name__ == '__main__':

  random.seed(1)
  G = random_graph(30)
  # plot(G)
  # plt.show()
  # N = len(G)
  # for n in range(N):
  #   G.add_edge(n, n+N)
  hosts = get_attr_nodes(G, 'type', 'host')
  # plt.show()
  S = random.sample(hosts, 6)
  r = S.pop()

  odpath = {}
  oddist = np.zeros([len(G), len(G)], dtype=np.int32)
  for o, d_p in nx.all_pairs_dijkstra_path(G):
    for d, p in d_p.items():
      odpath[o,d] = p.copy()
      p.reverse()
      odpath[d,o] = p
      if o == d:
        oddist[o,d] = 0
      else:
        oddist[o,d] = oddist[d,o] = len(p) - 1
  # nx.draw(G, with_labels=True)
  # plt.show()
  nx.set_node_attributes(G, 1, 'capacity')
  capacity = nx.get_node_attributes(G, 'capacity')


  paggrs = setup(S, r, odpath)
  m = 5

  ncombs = math.comb(len(paggrs), m)
  print("this turn:", len(paggrs), "try:", ncombs)
  Sarr = np.array(S, dtype=np.int32)
  start = time.perf_counter()
  for _ in range(2):
    for A in itertools.combinations(paggrs, m):
      P = dict(extract_paths(Sarr, np.array([r]+list(A), dtype=np.int32), oddist))
      T = nx.DiGraph()
      for s, t in P.items():
        p = odpath[s, t]
        for e in pairwise(p):
          T.add_edge(*e)
      #{plot(T)
      pass

  end = time.perf_counter()
  print(end-start)

