import networkx as nx
from functools import reduce
from collections import defaultdict
import numpy as np
try:
  from ...util import plot
except:
  import sys,os
  sys.path.append(os.getcwd())
  from inc.util import plot

def hamming_distance(a, b):
  count = 0
  for i, j in zip(a, b):
    if i != j:
      count += 1
  return count

def groupby(key, seq):
  # The reason for ... or grp in the lambda is that for this reduce() to work,
  # the lambda needs to return its first argument;
  # because list.append() always returns None the or will always return grp.
  # I.e. it's a hack to get around python's restriction that a lambda can only evaluate a single expression.
 return reduce(lambda grp, val: grp[key(val)].append(val) or grp, seq, defaultdict(list))

def irs_based(G: nx.Graph, S, r, k, DEBUG=False) -> nx.DiGraph:
  '''Specially for finding aggregation tree in Bcube
  k is the max level of switch
  '''
  assert G.graph['name'] == 'bcube'
  aggr_graph = nx.DiGraph()
  pos2server = {G.nodes[n]['pos']: n for n in G if G.nodes[n]['type']=='host'}
  pos2switch = {G.nodes[n]['pos']: n for n in G if G.nodes[n]['type']=='switch'}
  Spos = [G.nodes[s]['pos'] for s in S]
  rpos = G.nodes[r]['pos']
  stage_server = get_stage_server(Spos, rpos)
  if DEBUG:
    print(f"sources: {Spos}")
    print(f"target: {rpos}")
    print("stage-servers: ", stage_server)
    print()
  # stage_server[0].append(rpos)
  stage_num = next(iter(stage_server)) # max stage number

  routing_symbol = [False] * (stage_num + 1) # the routing symbol
  # * aggregate stage j's servers into stage j-1
  for j in range(stage_num, 0, -1): # stage_num..1
    servers = stage_server[j]
    if DEBUG:
      print(f"stage {j}'s servers: ", servers)
    min_group = np.iinfo(np.int32).max
    chosen_index = -1
    chosen_server_aggr = None
    chosen_new_servers = None
    # * servers are neighbours in dimension i
    for i in range(k + 1): # only dimension i is different
      if routing_symbol[i]: # * after a stage, a deminsion will be the same
        continue
      sender_aggr = {}
      for server in servers:
        aggr = list(server)
        aggr[i] = rpos[i]
        if server != tuple(aggr):
          sender_aggr[server] = tuple(aggr)
        else: # ! aggregate to itself which is not allowed
          Found = False
          for s in stage_server[j-1]: # check if there is a neighbor in next stage
            if hamming_distance(server, s) == 1:
              sender_aggr[server] = s
              Found = True
              break
          if not Found: # choose another deminsion to next stage
            for xx in range(k + 1):
              if aggr[xx] != rpos[xx]:
                aggr[xx] = rpos[xx]
                sender_aggr[server] = tuple(aggr)
                break
      # ! aggr server may already in the stage j-1, avoid counting twice
      new_servers = []
      for s, a in sender_aggr.items():
        if a not in stage_server[j-1] and a not in new_servers:
          new_servers.append(a)

      if len(new_servers) + len(stage_server[j-1]) < min_group:
        min_group = len(new_servers) + len(stage_server[j-1])
        chosen_index = i
        chosen_server_aggr = sender_aggr
        chosen_new_servers = new_servers
    # * end
    assert chosen_index != -1
    routing_symbol[chosen_index] = True # * this dimension now is the same with root server
    if DEBUG:
      print(f"choose routing {chosen_index}, aggr servers: ", list(chosen_server_aggr.values()))
      print("new servers: ", chosen_new_servers)
      print()
    # * add edge to graph
    for sender, aggr in chosen_server_aggr.items():
      aggr_switch = get_connected_switch(sender, chosen_index)
      aggr_graph.add_edge(pos2server[sender], pos2switch[aggr_switch])
      aggr_graph.add_edge(pos2switch[aggr_switch], pos2server[aggr]) # may be add multiple times but its ok

    # * add new servers to next stage
    stage_server[j-1].extend(chosen_new_servers)
  last_stage = stage_server[0]
  assert rpos in last_stage
  for s in last_stage:
    assert hamming_distance(s, rpos) <= 1
  # ! incase servers are not aggregate in stage 0
  # if len(last_stage) > 1:
  #   pick_one = last_stage[1]
  #   for t in range(k+1):
  #     if pick_one[t] != rpos[t]:
  #       break
  #   switch = get_connected_switch(rpos, t)
  #   for s in last_stage[1:]:
  #     aggr_graph.add_edge(pos2server[s], pos2switch[switch])
  #   aggr_graph.add_edge(pos2switch[switch], pos2server[rpos])
  for n in aggr_graph:
    aggr_graph.nodes[n]['capacity'] = G.nodes[n]['capacity']
  # aggr_graph._node.update((n, d.copy()) for n, d in G.nodes().items() if n in aggr_graph)
  assert nx.is_tree(aggr_graph) and max(d for n, d in aggr_graph.out_degree()) <= 1
  return aggr_graph


def get_stage_server(S:list, r:tuple) -> dict:
  hop_server = {}
  for s in S:
    dist = hamming_distance(s, r)
    hop_server[dist] = hop_server.get(dist, []) + [s]
  hop_server = dict(sorted(hop_server.items(), key=lambda x:x[0], reverse=True))
  d = next(iter(hop_server)) # the max stage, d <= k+1
  for j in range(d, -1, -1):
    hop_server[j] = hop_server.get(j, [])
  hop_server[0] = [r]
  return hop_server

def get_connected_switch(s, i):
  s = list(s)
  s.pop(i)
  s.insert(0, i)
  return tuple(s)


def test_irs():
  k = 2
  G = bcube(k, 4)
  G, _ = get_reindexed_graph(G)
  hosts = get_attr_nodes(G, 'type', 'host')

  # S = [13, 17, 18, 19, 22]
  # r = 8
  # irs_based(G, S, r, 1)
  # print()
  S = random.sample(hosts, 11)
  r = S.pop()

  D = irs_based(G, S, r, k, True)
  D.graph['rankdir'] = 'BT'
  pos = nx.nx_agraph.graphviz_layout(D, prog='dot')
  plot(D, pos=pos)
  plt.show()

def test_basic():
  k = 1
  G = bcube(k, 4)
  # send server
  v5 =  (1, 1) #13
  v9 =  (1, 2) #17
  v10 = (2, 2) #18
  v11 = (3, 2) #19
  v14 = (2, 3) #22
  # root server
  v0 = (0, 0) #8

  S = [v5, v9, v10, v11, v14]
  v1 = (1, 0)
  v2 = (2, 0)
  v3 = (3, 0)
  v4 = (0, 1)
  v8 = (0, 2)
  v12 = (0, 3)

  hop_server = get_stage_server(S, v0)
  # move into get_stage_server
  # hop_server = dict(sorted(hop_server.items(), key=lambda x:x[0], reverse=True))
  d = next(iter(hop_server)) # the max stage, d<=k+1
  # for j in range(d, -1, -1):
  #   hop_server[j] = hop_server.get(j, [])

  routing_symbol = [0] * (d+1) # the routing symbol

  for j in range(d, 0, -1):
    servers = hop_server[j] + hop_server[j-1]
    min_group = np.iinfo(np.int32).max
    chosen_index = -1
    chosen_servers = []
    for i in range(k + 1):
      # only dimension i is different
      server_group = {}
      for s in servers:
        aggr_server = list(s)
        aggr_server[i] = v0[i]
        aggr_server = tuple(aggr_server)
        server_group[aggr_server] = server_group.get(aggr_server, []) + [s]
      # ! aggr server may already in the stage j-1, avoid counting twice
      new_servers = []
      for s in server_group:
        if s not in hop_server[j-1]:
          new_servers.append(s)

      if len(new_servers) + len(hop_server[j-1]) <= min_group:
        chosen_index = i
        min_group = len(new_servers) + len(hop_server[j-1])
        chosen_servers = new_servers
    # * finish trying demension
    if j == 2:
      assert chosen_servers == [v1, v2, v3]
    if j == 1:
      assert chosen_servers == [v0]
    print(f"stage {j} will aggregate at stage {j-1}'s servers: ", chosen_servers)
    routing_symbol[j] = chosen_index
    assert chosen_index != -1
    print("through switches:", [get_connected_switch(s, chosen_index) for s in chosen_servers])
    hop_server[j-1].extend(chosen_servers)

  # todo put everything in a loop and form a aggregation graph


if __name__ == "__main__":
  # test_basic()
  # print()
  import random
  random.seed(4181)
  import matplotlib.pyplot as plt
  import numpy as np
  from rich import print
  import sys, os
  sys.path.append(os.getcwd())

  from inc.topo.bcube import bcube
  from inc.topo.bcube import bcube
  from inc.util.utils import get_reindexed_graph, get_attr_nodes
  from inc.util.utils import plot

  test_irs()