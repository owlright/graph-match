import networkx as nx
from heapq import heappop, heappush
from collections import deque

try:
  from ..util import *
except:
  import sys,os
  sys.path.append(os.getcwd())
  from inc.util import *

def takashami_tree(G:nx.DiGraph, sources, target, weight=None) -> nx.DiGraph:
  T = nx.DiGraph()
  srcs = set(sources)
  visited = set([target])
  q = []
  while(srcs):
    for v in visited:
      try: # incase input G is not strongly connected
        length, path = nx.multi_source_dijkstra(G, srcs, v, weight=weight)
        heappush(q, (length, path))
      except nx.NetworkXNoPath:
        continue
    c, path = heappop(q)
    srcs.remove(path[0])
    T.add_edges_from(pairwise(path))
  terminals = set(sources+[target])
  assert terminals.issubset(set(T.nodes()))
  return T



if __name__ == "__main__":
  G = nx.DiGraph()
  e = [('C', 'D', 3),
      ('C', 'E', 2),
      ('E', 'D', 1),
      ('D', 'F', 4),
      ('E', 'F', 2),
      ('E', 'G', 3),
      ('F', 'G', 2),
      ('G', 'H', 2),
      ('F', 'H', 1)
      ]
  for u,v,w in e:
    G.add_edge(u,v,weight=w)

  layout = {
    'C':[0, 1],
    'E':[1, 0],
    'D':[1, 1],
    'G':[2, 0],
    'F':[2, 1],
    'H':[3, 0]
  }

  length, path = nx.multi_source_dijkstra(G, ['C', 'E'],'D')

  # for n in G.nodes():
    # print(f"To {n}: min length:{length[n]}, path:{path[n]}")
  c,T = takashami_tree(G, ['C','D','E'], 'H')

  plot(T,pos=layout)
  plt.show()