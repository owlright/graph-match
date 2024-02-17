import random
from ..topo import random_graph
from ..util.utils import *
import numpy as np
import networkx as nx
import igraph as ig


def get_table_format(parameter_len, baseline_len, algo_len):
    # res_col_len = 10
    # cap_col_len = 10
    # ntree_col_len = 8
    # nS_col_len = 8
    param_col_width = 6
    baseline_col_width = 20
    algo_col_width = 15
    table_border = (
        "+"
        + "-" * (parameter_len * param_col_width + parameter_len - 1)
        + "-" * baseline_len * (baseline_col_width + 1)
        + "-" * algo_len * (algo_col_width + 1)
        + "+"
    )
    table_header_format = (
        "|"
        + f"{{:^{param_col_width}s}}|" * parameter_len
        + f"{{:^{baseline_col_width}s}}|" * baseline_len
        + f"{{:^{algo_col_width}s}}|" * algo_len
    )
    table_format = (
        f"|{{:^{param_col_width}.2f}}|"
        + f"{{:^{param_col_width}.2f}}|"
        + f"{{:^{param_col_width}d}}|"
        + f"{{:^{param_col_width}d}}|"
        + f"{{:^{int(baseline_col_width/3)}d}}|" * baseline_len * 3
        + f"{{:^{int(algo_col_width/2)}d}}|" * algo_len * 2
    )
    return table_border, table_header_format, table_format


def get_treesolution(sources, targets, trees):
    assert len(sources) == len(targets) == len(trees)
    number_of_trees = len(trees)
    aggr_traffic = 0
    non_aggr_traffic = 0
    naggrs = 0
    for k in range(number_of_trees):
        G = trees[k]
        S = sources[k]
        r = targets[k]
        aggr_traffic += G.number_of_edges()
        non_aggr_traffic += sum([nx.shortest_path_length(G, s, r) for s in S])
        naggrs += len([n for n in G if G.in_degree[n] >= 2 and n != r])
    return aggr_traffic, non_aggr_traffic, naggrs


def get_input(topo):
    """_summary_

    Parameters
    ----------
    topo : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    number_of_trees = random.randint(20, 50)
    number_of_sources = random.randint(50, 100)
    if topo == "fattree":
        return (
            number_of_trees,
            number_of_sources,
            None,
        )  # ! fattree topology can not change anyway
    elif topo == "random":  # ! random topo can change each time
        N = random.randint(300, 400)
        G_origin = random_graph(N)
    elif topo == "torus":
        G_origin
    return number_of_trees, number_of_sources, G_origin


def get_paths(G):
    """Unify all graphs(node index), set node capacity, get od matrix

    Parameters
    ----------
    G_origin : nx.Graph
        original undirected network topo

    Returns
    -------
    odpath : dictionary
        (src, destionation): path list

    oddist : ndarry
        oddist[src, dst] is the distance between src and dest
    """
    odpath = {}
    oddist = np.empty([len(G), len(G)], dtype=np.int32)
    # odpath = dict()
    g = ig.Graph(n=len(G), edges=list(G.edges()))
    for o in range(len(G)):
        for d in range(o + 1, len(G)):
            p = g.get_shortest_paths(o, to=d)[0]
            odpath[o, d] = p.copy()
            p.reverse()
            odpath[d, o] = p
            oddist[o, d] = oddist[d, o] = len(p) - 1
    for o in range(len(G)):
        odpath[o, o] = []
        oddist[o, o] = 0
    # for o, d_p in nx.all_pairs_dijkstra_path(G):
    #     for d, p in d_p.items():
    #         odpath[o, d] = p.copy()
    #         p.reverse()
    #         odpath[d, o] = p
    #         if o == d:
    #             oddist[o, d] = 0
    #         else:
    #             oddist[o, d] = oddist[d, o] = len(p) - 1
    return odpath, oddist
