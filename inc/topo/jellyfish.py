import copy
import networkx as nx

import random
import sys
from collections import defaultdict

try:
    from ..util import *
except:
    import sys, os

    sys.path.append(os.getcwd())
    from inc.util import *


def jellyfish(n_racks, n_ports, n_interconnects) -> nx.Graph:
    N = n_racks
    k = n_ports
    r = n_interconnects
    n_servers = N * (k - r)
    n_switches = N

    G = nx.random_regular_graph(r, N)
    while not nx.is_connected(G):  # ! ugly but efficient enough as n goes up
        G = nx.random_regular_graph(r, N)
    G.add_nodes_from(G.nodes(), type="switch")
    G.graph["graph"] = G
    G.graph["n_ports"] = n_ports
    G.graph["n_hosts"] = n_servers
    G.graph["n_switches"] = n_switches

    # connect the k-r servers to each switch
    host_index = N
    for s in range(0, N):
        for h in range(k - r):
            G.add_node(host_index, type="host")
            G.add_edge(s, host_index, type="inner")
            host_index += 1

    pos = nx.spring_layout(G)
    for i in G:
        G.nodes[i]["pos"] = pos[i]

    return G


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # for i in range(1000):
    #   RNG_SEED = i
    #   print(i)
    t = jellyfish(10, 4, 3)

    plot(t)
