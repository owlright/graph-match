import operator
import itertools
import networkx as nx
import random
import cProfile
import pstats
from heapq import heappop, heappush
from itertools import count

try:
    from .color import *
    from .plotters import *
except ImportError:
    import sys, os

    sys.path.append(os.getcwd())
    from inc.util.color import *
    from inc.topo import *
    from inc.util.plotters import *

INT32_MAX = 2147483647
import warnings
import functools


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def crange(i, j):
    return range(i, j + 1)


def set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def select(l, i, index):
    return [(x[0], x[1]) for x in l if x[index] == i]


def get_attr_nodes(G, k, v, comp_fun=operator.eq) -> list:
    return [x for x, y in G.nodes(data=True) if comp_fun(y.get(k), v)]


def get_attr_edges(G, k, v) -> list:
    return [(x, y) for x, y, z in G.edges(data=True) if z.get(k) == v]


def label_node_depth_in_arborescence(G: nx.DiGraph, root) -> None:
    """give each node its depth in the tree"""
    # todo The algorithm cannot deal with outdegree >=2
    for n in G:
        assert G.out_degree(n) <= 1
    current_distance = 0
    current_layer = {root}
    visited = {root}

    # this is basically BFS, except that the current layer only stores the nodes at
    # current_distance from source at each iteration
    while current_layer:
        next_layer = set()
        for node in current_layer:
            for child in G.predecessors(node):
                if child not in visited:
                    visited.add(child)
                    next_layer.add(child)
        G.add_nodes_from(current_layer, level=current_distance)
        current_layer = next_layer
        current_distance += 1
    G.graph["depth"] = current_distance - 1


def arborescence(G: nx.Graph, sources: list, root: int) -> nx.DiGraph:
    """Make an undirected tree to directed"""
    # terminals = get_attr_nodes(G, 'terminal', True)
    # terminals.remove(root)
    # tree_edges = get_attr_edges(G, 'intree', True)

    D = G.to_directed()
    for s in sources:  # remove the opposite edge
        D.remove_edges_from(list(pairwise(nx.algorithms.shortest_path(G, root, s))))

    assert is_arborescence(D)
    return D


# * the nx.is_arborescence direction is from root to source nodes
# * here is from source nodes to root
def is_arborescence(G: nx.DiGraph):
    return max(d for _, d in G.out_degree()) <= 1


def update_graph(G: nx.Graph, S: nx.Graph) -> None:
    """copy what attrs G needs from S"""
    G._node.update((n, d.copy()) for n, d in S.nodes.items() if G.has_node(n))
    G.add_edges_from(
        (u, v, d.copy()) for (u, v, d) in S.edges(data=True) if G.has_edge(u, v)
    )
    G.graph.update(S.graph)


def update_nodes_from(T, G, *node_attr):
    for n in T:
        for attr in node_attr:
            if attr in G.nodes[n]:
                T.nodes[n][attr] = G.nodes[n][attr]


def set_load(G: nx.Graph, connected_switches) -> None:
    # terminals = get_attr_nodes(G, 'type', 'host')
    for v in G.nodes():
        if v in connected_switches:
            # NOTE the terminal should aggregation its messages first \
            # so the load should always be 1?
            G.nodes[v]["load"] = random.randint(1, 10)
        else:
            G.nodes[v]["load"] = 0

    # leaf_switches = get_attr_nodes(G, 'level', 1)
    # for v in leaf_switches:
    #   # leaf switches gather its connected terminals' load
    #   total_load = [G.nodes[u]['load'] for u in G.neighbors(v) if G.nodes[u]['level']==0]
    #   G.nodes[v]['load'] = sum(total_load)


def set_capacity(G: nx.Graph, switch_memory, switches=None):
    all_switches = get_attr_nodes(G, "type", "switch")
    if switches is None:
        switches = all_switches
    else:
        left_switches = list(set(all_switches) - set(switches))
        G.add_nodes_from(left_switches, capacity=0)
    hosts = get_attr_nodes(G, "type", "host")
    G.add_nodes_from(hosts, capacity=0)
    G.add_nodes_from(switches, capacity=switch_memory)
    # num_switches = len(switches)
    # for _ in range(max_memory):
    #   switch = switches[random.randint(0, num_switches-1)]
    #   G.nodes[switch]['capacity'] += 1


def load(G: nx.DiGraph, v):
    """collect the number of msgs node v receive,
    but actually only the leaf switch has msgs
    """
    return G.nodes[v].get("load", 0)


def get_reindexed_graph(
    G: nx.Graph, hosts_num=None, switches_num=None
) -> tuple[nx.Graph, dict]:
    D = nx.Graph()
    indexMap = {}
    if not hosts_num:
        hosts_num = len(get_attr_nodes(G, "type", "host"))
    if not switches_num:
        switches_num = len(get_attr_nodes(G, "type", "switch"))
    hostIndex = switches_num
    switchIndex = 0
    for n in G:
        if G.nodes[n]["type"] == "switch":
            indexMap[n] = switchIndex
            D.add_node(switchIndex, type="switch")
            # if "pos" in G.nodes[n]:
            #   D.nodes[switchIndex]["pos"] = G.nodes[n]["pos"]
            switchIndex += 1
        else:
            indexMap[n] = hostIndex
            D.add_node(hostIndex, type="host")
            # if "pos" in G.nodes[n]:
            #   D.nodes[hostIndex]["pos"] = G.nodes[n]["pos"]
            hostIndex += 1
    # ! update node attrs
    D._node.update((indexMap.get(n, n), d.copy()) for n, d in G.nodes.items())
    D.graph.update(G.graph)
    D.add_edges_from(
        (indexMap.get(n1, n1), indexMap.get(n2, n2), d.copy())
        for (n1, n2, d) in G.edges(data=True)
    )
    # for u,v in G.edges:
    #   D.add_edge(indexMap[u], indexMap[v], weight=1)
    assert hostIndex == hosts_num + switches_num
    assert switchIndex == switches_num
    return D, indexMap


def do_cprofile(filename):
    """Decorator for function profiling."""

    def wrapper(func):

        def profiled_func(*args, **kwargs):
            # Flag for do profiling or not.
            DO_PROF = 1
            if DO_PROF:
                profile = cProfile.Profile()
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                # Sort stat by internal time.
                sortby = "tottime"
                ps = pstats.Stats(profile).sort_stats(sortby)
                ps.dump_stats(filename)
            else:
                result = func(*args, **kwargs)
            return result

        return profiled_func

    return wrapper


# def extract_paths_step(P, S, A, a, T, *args, Debug=False):
#   ''' Given decided Paths, source nodes, computation nodes, right_order and receiver node,
#   return the new generated Paths according to Prim algo.

#   Parameters
#   ----------
#   P : last step's P
#   S : sources
#   A : computation nodes, if this is empty then will return shortest paths
#   a : the node a is trying to add into A
#   T : the order of adding nodes after doing extract_paths(S, r, set(T))
#   args :
#   '''
#   # node2index = args[0]
#   # index2node = args[1]
#   assert not A[a] # a must not in A
#   joinNode = T[a]
#   # P[joinNode] = {}
#   if Debug:
#     print(f"The correct order is {T}")
#     print(f"Current A is {A} and trying to add {a}")
#   spm  = args[0]
#   Tlen = T.shape[0]

#   nodesBefore = np.full(Tlen, False, dtype=bool)
#   nodesBefore[:a] = A[:a]

#   # ! join_node's distance to the nodes before it
#   # ? do I really need to compare the distance ?
#   shortestNodeInd = np.argmin(spm[joinNode][T[nodesBefore]])
#   parentNode = T[nodesBefore][shortestNodeInd]
#   P[joinNode] = parentNode
#   # ! check if the nodes after the join_node can get a shorter distance to it
#   for i in range(a+1, Tlen):
#     node = T[i]
#     if A[i] and spm[node][joinNode] < spm[node][P[node]]:
#       P[node] = joinNode

#   # * get the paths from sources to target
#   for s in S:
#     if spm[s][joinNode] < spm[s][P[s]]:
#       P[s] = joinNode
#       # P[s][1] = spm[s][joinNode]
#   return P

# def extract_paths(S, r, A, oddist):
#   ''' Given decided Paths, source nodes, computation nodes and receiver node,
#   return the new generated Paths according to Prim algo.

#   Parameters
#   ----------
#   S : sources
#   r : the receiver/target
#   A : computation nodes, if this is empty then will return shortest paths
#   args : pass the info about node2index, shortest path and distance matrix
#         , because use nx.shortest will consume too much time
#   '''
#   P = {}

#   # T = A + [r]
#   # NUM_EDGES = G.number_of_edges()
#   # ! Step 1. extract the paths form computation nodes to target
#   # T = []
#   mstNodes = [r] + A
#   Tlen = len(mstNodes) # number of nodes in the mstT
#   # node2T = {}
#   # T2node = np.empty(Tlen, dtype=int)
#   TSeq = np.arange(Tlen, dtype=int)
#   # for i in range(Tlen):
#   #   node = mstNodes[i]
#   #   # node2T[node] = i
#   #   T2node[i] = node

#   # * similar to Prim algo
#   outMstSet = np.full(Tlen, True, dtype=bool)
#   parent = np.empty(Tlen, dtype=int)
#   dist = np.full(Tlen, INT32_MAX, dtype=int)
#   dist[0] = 0
#   parent[0] = -1
#   for _ in range(Tlen):
#     # * get the closet node to the tree
#     uind = np.argmin(dist[outMstSet])
#     uT = TSeq[outMstSet][uind]
#     u = mstNodes[uT] # ! the current closest node is u
#     outMstSet[uT] = False
#     # T.append(u)

#     # * connect the closest path to T
#     if u != r:
#       node_in_tree = mstNodes[parent[uT]]
#       # path = allpairspaths[uNode][node_in_tree]
#       P[u]= node_in_tree
#       # P[u][1] = spm[u][node_in_tree]

#     # * update the rest nodes' distance to T
#     # uInd = T2ind[uT]
#     for i in range(Tlen): # * update the rest distance
#       v = mstNodes[i]
#       if outMstSet[i] and oddist[v][u] < dist[i]:
#         dist[i] = oddist[v, u]
#         parent[i] = uT
#   # assert len(T) == Tlen # must done with all aggr nodes
#   # ! Step 2. extract the paths from sources to computation nodes or target
#   rowIndexes = np.empty([len(S), Tlen], dtype=int)
#   colIndexes = np.empty_like(rowIndexes, dtype=int)
#   for i, s in enumerate(S):
#     # rowIndexes = np.empty([1, len(T)], dtype=int)
#     # colIndexes = np.empty_like(rowIndexes, dtype=int)
#     rowIndexes[i, :] = s
#     for j in range(Tlen):
#       colIndexes[:, j] = mstNodes[j]

#   SV = oddist[rowIndexes, colIndexes]
#   ind = np.argmin(SV, axis=1)
#   for i, s in enumerate(S):
#     end_node = mstNodes[ind[i]]
#     # path = allpairspaths[s][end_node]
#     P[s] = end_node
#     # P[s][1] = spm[s, end_node]
#   return P

# def build_shortest_path_tree(G, S, r):
#   G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)
#   weight = lambda u,v,e: None
#   push = heappush
#   pop = heappop
#   dist = {}  # dictionary of final distances
#   seen = {}
#   pred = {}
#   paths = {source: [source] for source in G}
#   # fringe is heapq with 3-tuples (distance,c,node)
#   # use the count c to avoid comparing nodes (may not be able to)
#   c = count()
#   fringe = []
#   nS = 0
#   # for source in sources:
#   #     seen[source] = 0
#   push(fringe, (0, next(c), r))
#   while fringe:
#     (d, _, v) = pop(fringe)
#     if v in dist:
#       continue  # already searched this node.
#     dist[v] = d
#     if v in S:
#       nS += 1
#       if nS == len(S):
#         break
#     for u, e in G_succ[v].items():
#       cost = weight(u, v, e) # ! the direction is from sources to r
#       if cost is None:
#         cost = 1
#       uv_dist = dist[v] + cost
#       if u in dist:
#         u_dist = dist[u]
#         if uv_dist < u_dist:
#           raise ValueError("Contradictory paths found:", "negative weights?")
#         elif pred is not None and uv_dist == u_dist:
#           pred[u].append(v)
#       elif u not in seen or uv_dist < seen[u]:
#         seen[u] = uv_dist
#         push(fringe, (uv_dist, next(c), u))
#         if paths is not None:
#           paths[u] = [u] + paths[v]
#         if pred is not None:
#           pred[u] = [v]
#       elif uv_dist == seen[u]:
#         if pred is not None:
#           pred[u].append(v)

#   # The optional predecessor and path dictionaries can be accessed
#   # by the caller via the pred and paths objects passed as arguments.

#   return paths


def dijkstra(G, source, target=None):
    """Uses Dijkstra's algorithm to find shortest weighted paths

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty iterable of nodes
        Starting nodes for paths. If this is just an iterable containing
        a single node, then all paths computed by this function will
        start from that node. If there are two or more nodes in this
        iterable, the computed paths may begin from any one of the start
        nodes.

    weight: function
        Function with (u, v, data) input that returns that edges weight

    pred: dict of lists, optional(default=None)
        dict to store a list of predecessors keyed by that node
        If None, predecessors are not stored.

    paths: dict, optional (default=None)
        dict to store the path list from source to each node, keyed by node.
        If None, paths are not stored.

    target : node label, optional
        Ending node for path. Search is halted when target is found.

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    Returns
    -------
    distance : dictionary
        A mapping from node to shortest distance to that node from one
        of the source nodes.

    Raises
    ------
    NodeNotFound
        If any of `sources` is not in `G`.

    Notes
    -----
    The optional predecessor and path dictionaries can be accessed by
    the caller through the original pred and paths objects passed
    as arguments. No need to explicitly return pred or paths.

    """
    G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)
    paths = {dest: [dest] for dest in G}
    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []

    # for source in sources:
    #   seen[source] = 0
    #   push(fringe, (0, next(c), source))
    push(fringe, (0, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        for u, e in G_succ[v].items():
            # cost = weight(v, u, e)
            # if cost is None:
            #   continue
            vu_dist = dist[v] + 1
            # if cutoff is not None:
            #   if vu_dist > cutoff:
            #     continue
            if u in dist:
                u_dist = dist[u]
                if vu_dist < u_dist:
                    raise ValueError("Contradictory paths found:", "negative weights?")
                # elif pred is not None and vu_dist == u_dist:
                #   pred[u].append(v)
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                # if pred is not None:
                #   pred[u] = [v]
            # elif vu_dist == seen[u]:
            #   if pred is not None:
            #     pred[u].append(v)

    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.
    if target is None:
        return dist, paths
    else:
        return dist[target], paths[target]


def test_update_graph():
    random.seed(23234)
    G = fattree(4, False)
    for n in G:
        G.nodes[n]["from"] = "G"
    for e in G.edges():
        G.edges[e]["from"] = "G"
    members = [0, 3, "out"]
    F = nx.empty_graph(members)
    F.add_edge(0, 16)  # existed edge
    F.add_edge(16, 32)  # not existed edge
    update_graph(F, G)

    assert F.nodes[0]["from"] == "G"
    assert F[0][16]["from"] == "G"
    assert F.nodes["out"] == {}
    assert F[16][32] == {}


if __name__ == "__main__":
    test_update_graph()
