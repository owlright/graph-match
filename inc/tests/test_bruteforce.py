import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import sys,os
sys.path.append(os.getcwd())
from inc.util import *
from inc.algorithms.bruteforce_exp import *
from inc.topo import random_graph



def do_test():
  bruteforce_exp(S, r, capacity, 5, oddist, odpath)

if __name__ == '__main__':

  random.seed(1)
  G = random_graph(500)
  # N = len(G)
  # for n in range(N):
  #   G.add_edge(n, n+N)
  hosts = get_attr_nodes(G, 'type', 'host')
  # plt.show()
  S = random.sample(hosts, 11)
  r = S.pop()

  odpath = {}
  oddist = np.empty([len(G), len(G)], dtype=np.int32)
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

  bruteforce_exp(S, r, capacity,3, oddist, odpath)
