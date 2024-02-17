"""This script test the paper 'SOAR: Minimizing Network Utilization with Bounded In-network Computing's DP algorithm"""

import networkx as nx
import numpy as np
import random
import sys
import os
import re
from heapq import heappop, heappush

try:
    from ..util import *
    from ..topo import fattree
    from ..algorithms import takashami_tree
except:
    import sys, os

    sys.path.append(os.getcwd())
    from inc.util import *
    from inc.topo import fattree
    from inc.algorithms import takashami_tree
colors = {"red": 0, "blue": 1}
BLUE = colors["blue"]
RED = colors["red"]


def _binary_aggregation_tree(nary, levels) -> nx.DiGraph:
    G = nx.DiGraph()
    total_nodes = (nary**levels - 1) // (nary - 1)
    G.add_nodes_from([i for i in range(total_nodes)], capacity=1, is_compution=False)
    excepts_last_level_nodes = total_nodes - nary ** (levels - 1)
    for i in range(excepts_last_level_nodes):
        for j in range(nary * i + 1, nary * (i + 1) + 1):
            G.add_edge(j, i, weight=1)
    for d in range(levels):
        for i in range(2**d - 1, 2 ** (d + 1) - 1):
            G.add_node(i, level=d + 1)
    G.add_node(total_nodes, level=0)  # the 'd' node
    G.add_edge(0, total_nodes, weight=1)
    G.graph["depth"] = levels
    return G


def rho(T_succ, v, l):
    parent = list(T_succ[v])  # rho(v, A^l_v)
    rho = 0  # The rho actually is the reciprocal of bandwidth and the bandwidth to itself is infinity
    # u = v
    count = 0
    while count < l:
        p = parent[0]
        rho += 1  # ! to speed up all weight are 1 by default , but T.edges[u, p]['weight'] is more precise
        parent = list(T_succ[p])
        # u = p
        count += 1
    return rho


def mCost(l, i, Y, X, color):
    if color == colors["blue"]:
        tmp = Y[l, 1 : i + 1, BLUE] + np.flip(X[1, :i])
        return tmp.min()
        # ! do not delete below code in comment.
        # ! below code are always correct. you may need this in the future
        # return min([Y[l][i - j][BLUE] + X[1, j] for j in range(i)],
        #            default=0)
    else:
        tmp = Y[l, : i + 1, RED] + np.flip(X[l + 1, : i + 1])
        return tmp.min()
        # ! do not delete this comment!!
        # return min([Y[l][i - j][RED] + X[l + 1, j] for j in range(i + 1)],
        #            default=0)


def mSplit(l, i, Y, X, color, Debug=False):
    if Debug:
        print(f"parent node to root/blue distance: {l}")
        print(f"current tree max blues: {i}")
    if color == colors["blue"]:
        a = [Y[l][i - j][colors["blue"]] for j in range(i)]
        b = [X[1, j] for j in range(i)]
        if Debug:
            print(f"{a}")
            print(f"{b}")
        allocation = np.argmin(
            [Y[l][i - j][colors["blue"]] + X[1, j] for j in range(i)]
        )
        if Debug:
            print(f"allocation should be {allocation}")
        return allocation
    else:
        a = [Y[l][i - j][colors["red"]] for j in range(i + 1)]
        b = [X[l + 1, j] for j in range(i + 1)]
        if Debug:
            print(f"{a}")
            print(f"{b}")
        allocation = np.argmin(
            [Y[l][i - j][colors["red"]] + X[l + 1, j] for j in range(i + 1)]
        )
        if Debug:
            print(f"allocation should be {allocation}")
        return allocation


# ! the only difference between here and the paper is the index of children
# ! which means the distance and the number k have the same meanning with the paper,
# ! but child 0 here means child 1


def soar_gather(T: nx.DiGraph, k, Debug=False) -> int:  # k is the number of blue nodes
    T_succ = T._succ
    T_pred = T._pred
    X = {}
    Y = {}
    L = nx.get_node_attributes(T, "load")
    capacity = nx.get_node_attributes(T, "capacity")
    children = {}
    # The graph has to be directed because each node must know its parent and children
    levels = T.graph["depth"]
    # for d in reversed(range(0, levels + 1)):  # from bottom to up
    for d in range(levels, -1, -1):
        # * from the deepest level, some load may be on higher level, but it's ok, it will be calculated later.
        nodes = [x for (x, y) in T.nodes(data=True) if y["level"] == d]
        for v in nodes:
            childs = list(T_pred[v])
            if len(childs) == 0:
                X[v] = np.zeros((d + 1, k + 1))
                for l in range(d + 1):  # D(v)
                    X[v][l][0] = rho(T_succ, v, l) * L[v]
                    for i in range(1, k + 1):
                        if capacity[v] > 0:
                            X[v][l][i] = rho(T_succ, v, l)
                        else:
                            X[v][l][i] = rho(T_succ, v, l) * L[v]
            else:
                Y[v] = np.zeros((len(childs), d + 1, k + 1, len(colors)))
                X[v] = np.zeros((d + 1, k + 1))
                children[v] = (
                    childs  # record the children for reverse use in soar-color
                )
                # T.add_node(v, children=childs)
                for m, n in enumerate(childs):
                    for l in range(d + 1):  # distance to destation or blue node
                        for i in range(k + 1):
                            if m == 0:
                                Y[v][m][l][i][BLUE] = (
                                    X[n][1][i - 1] + rho(T_succ, v, l)
                                    if i > 0
                                    else np.inf
                                )
                                Y[v][m][l][i][RED] = X[n][l + 1][
                                    i
                                ]  # ! not sure about this + rho(T, v, l)*T.nodes[v]['load']
                            else:
                                Y[v][m][l][i][BLUE] = (
                                    mCost(l, i, Y[v][m - 1, :, :, :], X[n], BLUE)
                                    if capacity[v] > 0 and i > 0
                                    else np.inf
                                )
                                Y[v][m][l][i][RED] = mCost(
                                    l, i, Y[v][m - 1, :, :, :], X[n], RED
                                )

                for l in range(d + 1):
                    for i in range(k + 1):
                        X[v][l][i] = min(
                            Y[v][len(childs) - 1][l][i][BLUE],
                            Y[v][len(childs) - 1][l][i][RED],
                        )
    root = get_attr_nodes(T, "level", 0)[0]
    # for v in children:
    #   T.nodes[v]["children"] = children[v]
    return X[root][0][k], X, Y, children


def soar_color(T: nx.DiGraph, k, X, Y, children, Debug=False):
    levels = T.graph["depth"]
    root = get_attr_nodes(T, "level", 1)[0]
    T.add_node(
        root, color=colors["red"], l=1, num_blues=k
    )  # start from the root switch to bottom switch
    for d in range(1, levels + 1):
        nodes = get_attr_nodes(T, "level", d)
        for v in nodes:
            nChildren = T.in_degree(v)
            if nChildren > 0:
                nB = T.nodes[v]["num_blues"]
                l = T.nodes[v]["l"]
                if Debug:
                    print(f"current node: {v} distance: {l} k: {nB}")
                    print(f"Y blue {T.nodes[v]['Y'][-1,l,nB,colors['blue']]}")
                    print(f"Y red {T.nodes[v]['Y'][-1,l,nB,colors['red']]}")
                # * decide this node color
                if (
                    Y[v][-1, l, nB, colors["blue"]] < Y[v][-1, l, nB, colors["red"]]
                    and T.nodes[v]["capacity"] > 0
                ):
                    # print('\n'+T.nodes[v]['Y'][-1,:,:,colors['red']])
                    T.add_node(v, color=colors["blue"])
                    if Debug:
                        print("current node is blue")
                    l = 0  # reset the distance and deduce the used one blue node
                else:
                    T.add_node(v, color=colors["red"])
                    if Debug:
                        print("current node is red")

                # * decide how many blue nodes each child can have
                for m, c in enumerate(
                    reversed(children[v])
                ):  # children in reverse order
                    m = nChildren - m - 1
                    if m == 0:
                        break
                    j = mSplit(
                        l, nB, Y[v][m - 1, :, :, :], X[c], T.nodes[v]["color"], Debug
                    )
                    if Debug:
                        print(f"consider {m}th child node {c}")
                        print(f"child node {c} has {j} blues")
                    T.add_node(c, l=l + 1, num_blues=j)
                    nB = nB - j
                assert m == 0
                # handle c1 lastly
                if T.nodes[v]["color"] == colors["blue"]:
                    if Debug:
                        print(f"child node {c} has {nB-1} blues")
                    T.add_node(children[v][0], num_blues=nB - 1, l=l + 1)
                else:
                    if Debug:
                        print(f"child node {c} has {nB} blues")
                    T.add_node(children[v][0], num_blues=nB, l=l + 1)
                if Debug:
                    print("=" * 30)
            else:  # leaf switches
                lefts = T.nodes[v]["num_blues"]
                if lefts > 0 and T.nodes[v]["capacity"] > 0:
                    T.add_node(v, color=BLUE)
                else:
                    T.add_node(v, color=RED)
    allocation = nx.get_node_attributes(T, "color")
    blue_nodes = [k for k, v in allocation.items() if v == 1]
    return blue_nodes


def try_soar(tree, capacity, M, K, Debug=False):

    traffic, X, Y, children = soar_gather(tree, M)
    if Debug:
        print(f"Total traffic is {traffic}")
    blue_nodes = soar_color(tree, M, X, Y, children)
    key_blues = {k: [] for k in range(K)}
    blue_keys = {}
    for node_name in blue_nodes:
        key, node = re.findall(r"\d+", node_name)
        key = int(key)
        node = int(node)
        key_blues[key].append(node)
        blue_keys[node] = blue_keys.get(node, []) + [key]
    if Debug:
        print(f"nodes chosen to be blue: [blue]{key_blues}")
        print(f"blue node has keys: [blue]{blue_keys}")
        print("Now checking capcity constraints...")
    invalid_nodes = []
    for node, keys in blue_keys.items():
        if len(keys) > capacity[node]:
            heappush(invalid_nodes, (capacity[node] - len(keys), node))
    if Debug:
        if invalid_nodes:
            print("[red]There are invalid nodes!")
    return traffic, key_blues, blue_keys, invalid_nodes, X, Y


def mtree_soar(sources, targets, capacity, M, graphs, oddist, *args, Debug=False):
    tree = nx.DiGraph()
    tree_succ = tree._succ
    targets_ = []
    K = len(graphs)
    maxM = sum([len(S) - 1 for S in sources])
    M = M if M < maxM else maxM
    # compose all trees
    for k, T in enumerate(graphs):
        for node in T:
            node_name = "k" + str(k) + "n" + str(node)
            tree.add_node(
                node_name, key=k, node=node, capacity=1 if capacity[node] > 0 else 0
            )
            if T.out_degree(node) == 0:
                targets_.append(node_name)
            if T.in_degree(node) == 0:
                tree.nodes[node_name]["load"] = 1
        for e in T.edges():
            u = "k" + str(k) + "n" + str(e[0])
            v = "k" + str(k) + "n" + str(e[1])
            tree.add_edge(u, v, weight=1)
    # * combine all trees
    tree.add_node(0, capacity=0)
    tree.add_node(1, capacity=0)
    for t in targets_:
        tree.add_edge(t, 0, weight=0)
    tree.add_edge(0, 1, weight=0)
    label_node_depth_in_arborescence(tree, 1)

    _, key_blues, blue_keys, invalid_nodes, X, Y = try_soar(tree, capacity, M, K)
    while invalid_nodes:
        if Debug:
            print(f"Find invaild nodes {invalid_nodes}")
        for overflow, node in invalid_nodes:
            # while invalid_nodes:
            # overflow, node = heappop(invalid_nodes)
            # if Debug:
            #   print(f'Find most invaild node {node}')
            overflow = -overflow
            if Debug:
                print(
                    f"Has {len(blue_keys[node])} keys on {node}, but capacity is {capacity[node]}, so overflow {overflow} memory "
                )
            # get each tree's aggr traffic
            # aggr_traffic = []
            q = []
            for k in blue_keys[node]:
                t = targets_[k]
                # tc = X[t][2][len(key_blues[k])] # get the key's tree traffic
                node_name = "k" + str(k) + "n" + str(node)
                bluel = tree.nodes[node_name]["l"]
                parent_node = list(tree_succ[node_name])[0]
                for _ in range(bluel - 1):
                    parent_node = list(tree_succ[parent_node])[0]
                    if parent_node == t:
                        break
                nobluel = (
                    bluel + tree.nodes[parent_node]["l"] if parent_node != t else bluel
                )
                # aggr_traffic.append(tc)
                add_traffic_if_no_this_blue = (
                    X[node_name][nobluel][len(key_blues[k])]
                    - X[node_name][bluel][len(key_blues[k])]
                )
                # reduced_tc = tc_if_no_this_blue - tc
                # ideal aggr / real aggr = (0, 1], 1 is best and can be achieved if blue nodes are enough
                heappush(
                    q, (add_traffic_if_no_this_blue, k)
                )  # todo why not do like my mat algorithm does?

            while (
                overflow
            ):  # ! set the most un-affected tree capcacity this node to zero
                _, key = heappop(q)
                tree.nodes["k" + str(key) + "n" + str(node)]["capacity"] = 0
                overflow -= 1
        if Debug:
            print("Rerun the soar algorithm.")
            # capacity = nx.get_node_attributes(tree, 'capacity')
        _, key_blues, blue_keys, invalid_nodes, X, Y = try_soar(tree, capacity, M, K)

    if Debug:
        print("If not adjust paths:", X[1][0][M])

    key_paths = {
        # k: get_branches(G, k, sources[k], targets[k], x, key_compution_nodes[k])
        k: extract_paths(sources[k], targets[k], key_blues[k], oddist)
        for k in range(K)
    }
    cost = 0
    for k in range(K):
        P = key_paths[k]
        sindexes = list(P.keys())
        tindexes = [P[s] for s in sindexes]
        c = np.sum(oddist[sindexes, tindexes])
        cost += c
    return cost, key_paths, key_blues


def case1():
    T = _binary_aggregation_tree(2, 3)
    T.add_node(3, load=2)
    T.add_node(4, load=6)
    T.add_node(5, load=5)
    T.add_node(6, load=4)

    soar_gather(T, 2, True)
    soar_color(T, 2, True)
    print(get_attr_nodes(T, "color", colors["blue"]))


def case2():
    random.seed(2333)
    print("-" * 10 + "fattree" + "-" * 10)
    G = fattree(6).to_directed()
    hosts = get_attr_nodes(G, "type", "host")
    switches = get_attr_nodes(G, "type", "switch")
    # number_of_experiments = 10
    number_of_keys = 2
    M = [10]
    sources = []
    targets = []
    for _ in range(number_of_keys):
        load = 4
        ss = terminals = random.sample(hosts, load + 1)
        target = terminals.pop()
        sources.append(ss)
        targets.append(target)
    set_capacity(G)
    capacity = nx.get_node_attributes(G, "capacity")
    # first let we try no computation nodes
    traffic = 0
    for srcs, target in zip(sources, targets):
        for s in srcs:
            l, p = nx.single_source_dijkstra(G, s, target)
            traffic += l
    print(f"Non-compressed: {traffic}")

    for m in M:
        print(f"Total Memory M: {m}")
        graphs = [takashami_tree(G, srcs, t) for srcs, t in zip(sources, targets)]
        traffic = mtree_soar(
            graphs, G.copy(), sources.copy(), targets.copy(), capacity.copy(), m
        )
        print(f"Algorith mat's result: {traffic}")
        print("-" * 80)


if __name__ == "__main__":
    case2()
