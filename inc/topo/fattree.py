import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

import random
import operator


def set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))


def fattree(k, singleHost=True):
    pods = k
    num_hosts = (pods**3) // 4
    num_edge_switches = num_agg_switches = pods * pods // 2
    num_core_switches = (pods * pods) // 4
    num_total_nodes = (
        num_hosts + num_edge_switches + num_agg_switches + num_core_switches
    )

    num_hosts_per_pod = (k // 2) ** 2
    num_edges_per_pod = num_aggrs_per_pod = k // 2
    num_groups = k // 2
    F = nx.Graph()
    F.graph["name"] = "fattree"
    F.add_nodes_from([(i, {"level": -1}) for i in range(0, num_total_nodes)])

    start_index = 0

    for i in range(start_index, num_hosts):
        F.nodes[i]["type"] = "host"
        F.nodes[i]["level"] = 0
        F.nodes[i]["pod"] = i // num_hosts_per_pod
        F.nodes[i]["pos"] = (i, 0)
        start_index += 1

    count = start_index
    # the first core switch x position is the middle of the first pod hosts' positions
    core_start_pos = mean([F.nodes[n]["pos"][0] for n in range(num_hosts_per_pod)])
    # left the same blank space at the end
    core_end_pos = F.nodes[num_hosts - 1]["pos"][0] - core_start_pos
    core_width = core_end_pos - core_start_pos
    core_space = core_width / (num_core_switches - 1)

    pod_ids = list(np.ravel([[i] * (k // 2) for i in range(k)]))
    for i in range(start_index, start_index + num_edge_switches):
        # F.nodes[i]['type'] = 'edge'
        F.nodes[i]["type"] = "switch"
        F.nodes[i]["level"] = 1
        F.nodes[i]["pod"] = pod_ids[i - count]
        hosts_per_edge = k // 2
        connected_hosts = [
            j
            for j in range(
                (i - count) * hosts_per_edge,
                (i - count) * hosts_per_edge + hosts_per_edge,
            )
        ]
        F.nodes[i]["pos"] = (mean([F.nodes[n]["pos"][0] for n in connected_hosts]), 4)
        for j in connected_hosts:
            F.add_edge(i, j)  # conncet the edge switch to hosts
        start_index += 1

    count = start_index
    for i in range(start_index, start_index + num_agg_switches):
        F.nodes[i]["type"] = "switch"
        F.nodes[i]["level"] = 2
        F.nodes[i]["pod"] = pod_ids[i - count]
        F.nodes[i]["group"] = (i - count) % num_aggrs_per_pod
        F.nodes[i]["pos"] = (F.nodes[i - num_edge_switches]["pos"][0], 8)
        for j in [
            x
            for (x, y) in F.nodes(data=True)
            if y["level"] == 1 and y["pod"] == F.nodes[i]["pod"]
        ]:
            F.add_edge(i, j)  # conncet the aggr switch to edge switches
        start_index += 1

    count = start_index
    # I'm not sure if we have to get sequenes like [0, 0, 1, 1]
    # [0, 1, 0, 1] may be good as well and can simply the code with '%'
    group_ids = list(np.ravel([[i] * (k // 2) for i in range(k // 2)]))
    for i in range(start_index, start_index + num_core_switches):
        F.nodes[i]["type"] = "switch"
        F.nodes[i]["level"] = 3
        F.nodes[i]["group"] = group_ids[i - count]
        F.nodes[i]["pos"] = (core_start_pos + core_space * (i - count), 12)
        for j in [
            x
            for (x, y) in F.nodes(data=True)
            if y["level"] == 2 and y["group"] == F.nodes[i]["group"]
        ]:
            F.add_edge(i, j)  # conncet the aggr switch to edge switches
        start_index += 1
    for u, v in F.edges():
        F.edges[u, v]["weight"] = 1

    if singleHost:
        for i in range(num_hosts):
            F.remove_node(i)
        leaf_switches = [x for x, y in F.nodes(data=True) if y.get("level") == 2]
        for i in range(len(leaf_switches)):
            ls = leaf_switches[i]
            F.add_node(i, pos=(F.nodes[ls]["pos"][0], 0), type="host", level=0)
            F.add_edge(i, ls)

    return F


if __name__ == "__main__":
    F = fattree(4)
    # give some random weights to test KSB algorithm
    for u, v in F.edges():
        F.edges[u, v]["weight"] = random.randint(0, 10)
    # for i in range(F.number_of_nodes()):
    #   print(F.nodes[i])

    fig, ax = plt.subplots()

    solution_graph = F.copy()
    terminal_nodes = [1, 3, 4, 11, 15]
