import networkx as nx
from heapq import heappop, heappush, heapify
from .tree.construct import construct_tree_from_paths
try:
  from ..util import *
except ImportError:
  import sys, os
  sys.path.append(os.getcwd())
  from util import *
  from topo import minitopo


def gst(sources, targets, steiner_trees, capacity, M, oddist, odpath):
  K = len(sources)
  As = [[] for r in targets]

  Ps = {}
  flow_paths = {}
  forbiddens = set([n for n,v in capacity.items() if v == 0])
  for k in range(K):
    S = sources[k]
    r = targets[k]
    T = steiner_trees[k]
    T_succ = T._succ
    Ps[k] = {}
    flow_paths[k] = {}
    for s in S:
      Ps[k][s] = r
      n = s
      path = []
      while n != r:
        path.append(n)
        n = list(T_succ[n])[0]
      path.append(r)
      flow_paths[k][s] = path
      forbiddens.add(s)
    # T = construct_tree_from_paths(Ps[k], odpath)
    forbiddens.add(r)

  candiateIncNodes = []
  deal_keys = list(range(K))
  while M != 0:
    for k in deal_keys:
      inflows = {}
      dests = {}
      flow_path = flow_paths[k]
      A = As[k]
      # find aggregation nodes
      for _, p in flow_path.items():
        for node in p:
          if node not in forbiddens and node not in A:
            inflows[node] = inflows.get(node, 0) + 1
            dests[node] = p[-1]
      bestNode = None
      bestCost = 0
      for node, flows_num in inflows.items():
        if flows_num >= 2:
          cost_can_reduced = flows_num * oddist[node, dests[node]]
          # cost_can_reduced = flows_num * (len(flow_path[node]) - 1)
          if cost_can_reduced > bestCost:
            bestCost = cost_can_reduced
            bestNode = node
      if bestNode is not None:
        heappush(candiateIncNodes, (-bestCost, k, bestNode))

    if not candiateIncNodes: # ! this will happen if M is too big that no more tree needs
      break
    _, key_to_change, aggr_to_add = heappop(candiateIncNodes)
    P = Ps[key_to_change]
    flow_path = flow_paths[key_to_change]
    As[key_to_change].append(aggr_to_add)
    affect_flows  = []
    # other_flows = []
    for _, p in flow_path.items():
      s = p[0]
      if aggr_to_add in p: # merge the path
        t = p[-1]
        P[s] = aggr_to_add
        P[aggr_to_add] = t
        index = p.index(aggr_to_add)
        affect_flows.append((s, index)) # cannot change flow_path here
      # else:
      #   other_flows.append(s)

    for s, index in affect_flows:
      flow_path[aggr_to_add] = flow_path[s][index:]
      flow_path[s] = flow_path[s][:index+1]

    deal_keys = [key_to_change] # ! only the changed key tree need to recalc it cost again
    capacity[aggr_to_add] -= 1
    candiateIncNodes_replace = []
    if capacity[aggr_to_add] == 0:
      forbiddens.add(aggr_to_add) # this node cannot be used anymore
      for item in candiateIncNodes:
        if item[2] in forbiddens:
          deal_keys.append(item[1])
        else:
          candiateIncNodes_replace.append(item)
      candiateIncNodes[:] = candiateIncNodes_replace
      heapify(candiateIncNodes)
    M = M - 1

  for k in range(K):
    P = Ps[k]
    A = As[k]
    S = sources[k]
    r = targets[k]
    for s in S: # ! we dont change the aggr flows direction
      # ! but the flow from sources should still be correct
      if P[s] == r:
        for a in A:
          if oddist[s, a] < oddist[s, P[s]]:
            P[s] = a
  # for k in range(K):
  #   T = steiner_trees[k]
  #   T_pred = T._pred
  #   r = targets[k]
  #   depth = 0
  #   waited = deque([(r, depth)])
  #   while waited:
  #     node, depth = waited.popleft()
  #     if node not in forbidden:
  #       indegree = len(T_pred[node])
  #       node_rank.append((-indegree, k, node))
  #       if indegree >= 2:
  #         depth = 0
  #       else:
  #         depth += 1
  #     for child in T_pred[node]:
  #       waited.append((child, depth+1))
  # random.shuffle(node_rank)
  # node_rank.sort()

  # # allnodes.sort()
  # used = 0
  # used_nodes = set()
  # last_costs = [INT32_MAX]*K

  # while True:
  #   old_used = used
  #   for _, key, node in node_rank:
  #     # each turn get the next best node
  #     if (key, node) in used_nodes:
  #       continue
  #     if used == M:
  #       break
  #     else:
  #       P = extract_paths(sources[key], targets[key], As[key]+[node], oddist)
  #       sindexes = list(P.keys())
  #       tindexes = [P[s] for s in sindexes]
  #       cost = np.sum(oddist[sindexes, tindexes])
  #     if capacity[node] >= 1 and cost < last_costs[key]:
  #       As[key].append(node)
  #       capacity[node] -= 1
  #       used += 1
  #       last_costs[key] = cost
  #       used_nodes.add((key, node))
  #   if used == old_used:
  #     # ! this will happen if M is used up or
  #     # ! too big that no more tree needs
  #     break


  # for k in range(K):
  #   Ps[k] = extract_paths(sources[k], targets[k], As[k], oddist)

  cost = 0
  for k in range(K):
    P = Ps[k]
    T = construct_tree_from_paths(P, odpath)
    nouse = []
    for n in As[k]:
      if T.in_degree(n) == 1:
        nouse.append(n)
    for n in nouse:
      As[k].remove(n)
    sindexes = list(P.keys())
    tindexes = [P[s] for s in sindexes]
    c = np.sum(oddist[sindexes, tindexes])
    cost += c
  # assert sum(last_costs)==cost
  return cost,Ps,As

if __name__=="__main__":
  G = minitopo()
  switches = get_attr_nodes(G, 'type', 'switch')
  hosts = get_attr_nodes(G, 'type', 'host')
  G.add_nodes_from(hosts, capacity=0)
  for v in switches:
    G.nodes[v]['capacity'] = 1
  sources = [['a','b','c','d','e','g','f','h']]
  targets = ['r']
  set_capacity(G, 10)
  capacity = nx.get_node_attributes(G, 'capacity')
  cost, flow_paths = gst(G, sources, targets, capacity, 4)
  for k,P in enumerate(flow_paths):
    draw_flow_paths(G, P)
    plt.show()


