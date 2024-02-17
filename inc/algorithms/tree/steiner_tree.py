import itertools
import networkx as nx
import numpy as np


def build_steiner_tree(
    S, r, oddist: np.array, odpath: dict, forbidden_nodes={}
) -> nx.DiGraph:
    """Given sources, target, all paths and dists between them
    return the new generated Paths according to Prim algo also known as Takashami.

    Parameters
    ----------
    S : sources
    r : the target
    oddist : od pair distance
    odpath : { source : destination }
    forbidden_nodes : these nodes cannot use as steiner nodes because they have no capacity

    Returns
    ----------
    nx.DiGraph
      a directed graph connect S and r with direction from S to r
    """
    T = [r] + S
    merged_tree = nx.DiGraph()
    Tlen = len(T)
    outMstSet = np.full(Tlen, True, dtype=bool)
    parent = np.empty(Tlen, dtype=int)
    # merged_node_owner = np.empty_like(merged_node, dtype=int)
    dist = np.full(Tlen, np.iinfo(np.int).max, dtype=int)
    TSeq = np.arange(Tlen, dtype=int)
    dist[0] = 0

    for _ in range(Tlen):
        uind = np.argmin(dist[outMstSet])
        uT = TSeq[outMstSet][uind]
        u = T[uT]
        outMstSet[uT] = False

        if u != r:
            node_in_tree = parent[uT]
            add_path = odpath[u, node_in_tree]
            for e in itertools.pairwise(add_path):
                merged_tree.add_edge(*e)
        else:
            assert u == r
            add_path = [r]
        # * update the outside-tree nodes' distances
        for i in range(Tlen):
            if outMstSet[i]:
                v = T[i]
                closet_dist = dist[i]
                closet_parent = parent[i]
                for new_node in add_path:
                    if new_node not in forbidden_nodes or new_node == r:
                        if oddist[v, new_node] < closet_dist:
                            closet_dist = oddist[v, new_node]
                            closet_parent = new_node
                dist[i] = closet_dist
                parent[i] = closet_parent

    assert nx.is_tree(merged_tree) and max(d for n, d in merged_tree.out_degree()) <= 1
    return merged_tree


if __name__ == "__main__":
    import random
    import sys, os

    sys.path.append(os.getcwd())
    from inc.exps import *
    from inc.util import *
    from inc.topo import *

    random.seed(322)
    G = bcube(3, 4)
    G, _ = get_reindexed_graph(G)
    odpath, oddist = get_paths(G)
    hosts = get_attr_nodes(G, "type", "host")
    S = random.sample(hosts, 12)
    r = S.pop()
    g = build_steiner_tree(S, r, oddist, odpath)
    plot(g)
    plt.savefig("tree.png")
