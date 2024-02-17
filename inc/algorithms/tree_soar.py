import sys, os

sys.path.append(os.getcwd())
from algorithms import *
from util import *
import networkx as nx

import logging

logger = logging.getLogger(__name__)


def get_tree(
    G: nx.DiGraph, sources: list, root: int, algorithm="avalanche", weight=None
):
    assert G.is_directed()
    if algorithm == "avalanche":
        T = avrouter(G.copy(), sources.copy(), root, weight)
    elif algorithm == "kmb":
        # networkx use kmb algorithm as default
        T = kmbtree(G, sources.copy(), root, weight)
    elif algorithm == "shortest":  # also named MPH
        T = sptree(G, sources.copy(), root, weight)
    elif algorithm == "pruned_dijistra":
        T = pruned_dijistra_directed_tree(G, sources.copy(), root, weight)
    else:
        raise KeyError
    return T


def tree_soar(
    G: nx.DiGraph, sources: list, target: int, L, M, algorithm="avalanche", weight=None
):
    """
    Parameters
    ----------
    L : data chunk's key size
    M : total memory job can use
    """
    results = {}
    traffic = 0
    # * source terminals load are already summed on their connected leaf switches
    # * and other hosts cannot be forwarding or computing nodes
    # * I don't need the hosts, because the first edge is already known
    # todo: the last edge is known too, but to compatible with soar, I keep it
    # unnecessary_nodes = [u for u in get_attr_nodes(G, 'type', "host") if u != root]
    # some hosts connect to the same switch, so need to put them in set first
    # source_switches = list(set([next(G.neighbors(u)) for u in sources]))
    # logger.debug(f"source_siwtches: {source_switches}")
    # root_switch = next(G.neighbors(root))

    # * this part is for ilp algorithm
    if algorithm == "ilp":
        traffic, results = ilp_solve(G, sources, target, M, L)
        return traffic, results

    # multiple hosts may connected to the same switch
    source_switches = list(set([next(G.successors(u)) for u in sources]))
    # target_switch = next(G.predecessors(target))
    for u in source_switches:  # sum up all source hosts load on the leaf swtich
        G.nodes[u]["load"] = sum(
            [
                G.nodes[i]["load"]
                for i in G.predecessors(u)
                if G.nodes[i]["type"] == "host"
            ]
        )
        logger.info(f"{u} load: {G.nodes[u]['load']}")
    # * this part is for two-step heuristic algorithms
    # * allocate memory equally to each key
    aggr_tree = get_tree(G, source_switches, target, algorithm, weight)

    label_node_depth_in_arborescence(aggr_tree, target)
    equal_share = M // L
    left_share = M % L
    share = [equal_share] * L  # * L is the number of times to run soar
    for i in range(left_share):  # * divide the left evenly
        share[i] += 1

    for i in range(L):
        # print(f"key {i}")
        traffic += soar_gather(aggr_tree, share[i])
        blue_nodes = soar_color(aggr_tree, share[i])
        need_recalc = False
        for u in blue_nodes:  # ! must remove the used memory
            results[u] = (
                results.get(u, 0) + 1
            )  # switch u allocate 1 unit memory to this key
            G.nodes[u]["capacity"] -= 1
            if G.nodes[u]["capacity"] == 0:
                logger.debug(f"{u} is exhaust")
                need_recalc = (
                    True  # TODO for now, only when the capacity==0 then recalc the tree
                )
                # G.remove_node(u)
                # plot(G)
        if need_recalc:
            aggr_tree = get_tree(G, sources, target, algorithm)
            label_node_depth_in_arborescence(aggr_tree, root)

    return traffic, results


if __name__ == "__main__":
    ...
