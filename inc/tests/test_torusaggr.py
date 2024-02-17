import networkx as nx
import random
import matplotlib.pyplot as plt
import math

import sys, os

sys.path.append(os.getcwd())
from inc.util import get_reindexed_graph, get_attr_nodes
from inc.topo import torus
from inc.algorithms import camdoop
from inc.util.plotters import plot

if __name__ == "__main__":
    random.seed(1454)
    G = torus(3)
    G_reindexed, _ = get_reindexed_graph(G)
    del G
    hosts = get_attr_nodes(G_reindexed, "type", "host")
    S = random.sample(hosts, 21)
    r = S.pop()
    for node in S + [r]:
        switch = next(G_reindexed.neighbors(node))
        print(G_reindexed.nodes[switch]["pos"])
    a = camdoop(G_reindexed, S, r, "x:z:y")
    assert nx.is_tree(a)
    G_reindexed.graph["rankdir"] = "BT"
    pos = nx.nx_agraph.graphviz_layout(G_reindexed, prog="dot")
    nx.draw(a, labels=dict(a.nodes(data="pos", default=0)), pos=pos)
    plt.show()
