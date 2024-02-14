import networkx as nx
import itertools
import matplotlib.pyplot as plt

def bcube(k=0, n=4) -> nx.Graph:
  '''BCube is a recursively defined structure. k also means level, n also means n-port switch
  A BCube_k is constructed from n Bcube_k-1 and n^k n-port switches
  n^(k+1) servers
  each node addres is a (sk,sk-1,...,0), length is k+1
  '''
  def pos2index(pos):
    # ! use little endian
    index = 0
    for i, p in enumerate(pos):
      index += p*n**i
    return index

  G = nx.empty_graph()
  G.graph['name'] = "bcube"
  for level in range(k+1):
    for index in itertools.product(range(n), repeat=k):
      switch_node = (level, *index)
      G.add_node(switch_node, type='switch', pos=switch_node)

      for i in range(n):
        ss = list(index)
        # * insert at position: level
        server_node = tuple(ss[:level] + [i] + ss[level:])
        G.add_node(pos2index(server_node), type='host', pos=server_node)
        G.add_edge(pos2index(server_node), switch_node)
  return G

def test_big_number():
  G = bcube(3, 6)

if __name__ == "__main__":
  test_big_number()
