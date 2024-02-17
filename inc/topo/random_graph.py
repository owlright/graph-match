import networkx as nx
import random
import math


def random_graph(N: int, H=-1, p=-1) -> nx.Graph:
    if p < 0:
        p = 2 * math.log(N) / N
    if H < 0:
        H = N

    er = nx.erdos_renyi_graph(N, p)
    while not nx.is_connected(er):  # ! incase the random graph is not connected
        er = nx.erdos_renyi_graph(N, p)
    host_connected_switches = random.sample(er.nodes(), H)
    switches = list(er)
    for n in switches:
        er.nodes[n]["type"] = "switch"
        if n in host_connected_switches:
            host_id = n + N
            er.add_node(host_id, type="host")
            er.add_edge(n, n + N)
    return er


if __name__ == "__main__":
    G = random_graph(100, 25)
