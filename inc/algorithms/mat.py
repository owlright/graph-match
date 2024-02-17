import networkx as nx
import random
from heapq import heappop, heappush, heapify
from collections import deque
import sys, os
import numpy as np

try:
    from ..util.utils import *  # ? if use 'from ..util import *' pylance will report not defined symbol
    from .extract_paths import *
    from ..exps.exp_setup import get_paths
except ImportError:
    import sys, os

    sys.path.append(os.getcwd())
    from inc.util.utils import *
    from inc.topo import fattree, random_graph
    from inc.algorithms.extract_paths import *
    from inc.exps.exp_setup import get_paths
    from inc.algorithms.tree.construct import construct_tree_from_paths
    import random

INT32_MAX = 2147483647


def update_resource(aggr_to_add, capacity, candiateIncNodes, forbiddens) -> list:
    """Update the aggr_to_add's resource, when aggr_to_add has no resource,
    return the affected keys
    """
    capacity[aggr_to_add] -= 1
    candiateIncNodes_replace = []
    deal_keys = []
    if capacity[aggr_to_add] == 0:
        forbiddens.add(aggr_to_add)  # this node cannot be used anymore
    for item in candiateIncNodes:
        if item[2] in forbiddens:
            deal_keys.append(item[1])
        else:
            candiateIncNodes_replace.append(item)
    candiateIncNodes[:] = candiateIncNodes_replace
    heapify(candiateIncNodes)
    return deal_keys


def gstep(sources, targets, capacity, M, oddist: np.ndarray, odpath, Debug=False):
    K = len(sources)
    As = {k: [] for k in range(K)}  # selected aggregation nodes
    Ps = {k: {} for k in range(K)}  # paths of aggregation tree/flows
    Ws = {k: set() for k in range(K)}  # potential aggregation nodes

    forbiddens = set()  # * nodes are exhausted
    last_costs = [0] * K
    # ! remove nodes cannot be computation nodes
    # include all sources and targets here
    for node in capacity:
        if capacity[node] == 0:
            forbiddens.add(node)

    for k in range(K):
        S = sources[k]
        r = targets[k]
        for s in S:
            Ps[k][s] = r  # * get intial flow paths
            for n in odpath[s, r]:
                if n not in forbiddens:
                    Ws[k].add(n)  # ! get potential nodes W for each tree
        last_costs[k] = np.sum(oddist[S, r])

    # ! begin to choose good aggr nodes as inc nodes
    candiate_aggrs = []
    deal_keys = list(range(K))
    while M != 0:
        for k in deal_keys:
            S = sources[k]
            Slen = len(S)
            r = targets[k]
            A = As[k]
            P = Ps[k]
            last_cost = last_costs[k]

            minCost = 0
            minNode = None  # ! if nodes in T are all used, minNode will still be None after the loop below
            minP = None
            mstnodes = [r] + A
            mstlen = len(mstnodes) + 1
            for a in Ws[k]:  # try each potential aggr nodes
                mst = mstnodes + [a]
                parr = extract_paths_njit_nodict(
                    np.array(S, dtype=np.int32), np.array(mst, dtype=np.int32), oddist
                )
                i = 0
                pdict = {}
                cost = 0
                for i in range(1, Slen + mstlen):
                    t = parr[i]
                    if i < mstlen:
                        s = mst[i]
                    else:
                        s = S[i - mstlen]
                    pdict[s] = parr[i]
                    cost += oddist[s, t]
                if cost - last_cost < minCost:
                    minCost = cost - last_cost
                    minNode = a
                    minP = pdict
            # !! node may be 0, you have to write not None here
            if (
                minNode is not None
            ):  # * the best inc node that can reduce most cost of kth tree
                heappush(candiate_aggrs, (minCost, k, minNode, minP))
            else:  # * this tree doesnt need aggregation nodes any more
                continue

        if (
            not candiate_aggrs
        ):  # ! this will happen if M is too big that no more tree needs
            break
        reducedCost, key_to_change, aggr_to_add, paths_to_replace = heappop(
            candiate_aggrs
        )
        # tree = construct_tree_from_paths(paths_to_replace, odpath)
        if Debug:
            print(f"choose key {key_to_change} set aggr {aggr_to_add}")
        # * update the chosen key's Paths
        for s, t in paths_to_replace.items():
            for n in odpath[s, t]:
                if n not in forbiddens and n not in As[key_to_change]:
                    Ws[key_to_change].add(n)  # ! find more potential aggregation nodes
        Ws[key_to_change].remove(aggr_to_add)
        Ps[key_to_change] = paths_to_replace
        As[key_to_change].append(aggr_to_add)
        last_costs[key_to_change] = last_costs[key_to_change] + reducedCost
        # T = construct_tree_from_paths(Ps[key_to_change] , odpath)

        deal_keys = [
            key_to_change
        ]  # ! only the changed key tree need to recalc it cost again
        # * update the resource use
        affect_keys = update_resource(aggr_to_add, capacity, candiate_aggrs, forbiddens)
        deal_keys = deal_keys + affect_keys
        M = M - 1
    cost = 0
    for k in range(K):
        P = Ps[k]
        # T = construct_tree_from_paths(P, odpath)
        sindexes = list(P.keys())
        tindexes = [P[s] for s in sindexes]
        c = np.sum(oddist[sindexes, tindexes])
        cost += c

    return cost, Ps, As


def mat(sources, targets, steiner_trees, capacity, M, oddist, odpath, Debug=False):
    K = len(sources)
    As = [[] for r in targets]
    Ps = {}
    potential_aggrs = []
    forbiddens = set()
    last_costs = [0] * K
    # ! remove nodes cannot be computation nodes
    for k in range(K):
        S = sources[k]
        r = targets[k]
        for s in S:
            forbiddens.add(s)
        forbiddens.add(r)
    for node in capacity:
        if capacity[node] == 0:
            forbiddens.add(node)
    # ! get potential nodes W for each tree
    for k in range(K):
        S = sources[k]
        r = targets[k]
        Ps[k] = {}
        paggrs = []
        for s in S:
            Ps[k][s] = r
            for n in odpath[s, r]:
                if n not in forbiddens:
                    paggrs.append(n)

        potential_aggrs.append(set(paggrs))
        cost = np.sum(oddist[S, r])
        last_costs[k] = cost

    # ! begin to choose good aggr nodes as inc nodes
    candiateIncNodes = []
    deal_keys = list(range(K))
    while M != 0:
        for k in deal_keys:
            S = sources[k]
            r = targets[k]
            A = As[k]
            P = Ps[k]
            last_cost = last_costs[k]

            minCost = 0
            minNode = None  # ! if nodes in T are all used, minNode will still be None after the below loop
            minP = None
            intree = [r] + A
            for a in potential_aggrs[k]:  # try each potential aggr nodes
                tmp = P.copy()
                dists = []
                for i in intree:
                    dists.append((oddist[a, i], i))  # * distance between node and tree
                dists.sort()
                # parent_node_index = np.argmin(oddist[a, intree])
                # parent_node = intree[parent_node_index]
                joint_node = dists[0][1]
                visited = set()
                for _, n in dists:
                    if n in visited:
                        continue
                    # ! check if joint node passes a
                    is_pred = False
                    next_node = n
                    while next_node != r:
                        visited.add(next_node)
                        if next_node == joint_node:
                            is_pred = True
                            break
                        else:
                            next_node = tmp[next_node]
                    if not is_pred:
                        second_joint_node = n
                        break

                succ_nodes = set()
                if joint_node != r:

                    to_tree_direction = (
                        oddist[a, joint_node] + oddist[joint_node, P[joint_node]]
                    )

                    from_tree_direction = (
                        oddist[joint_node, a] + oddist[a, second_joint_node]
                    )
                    change_direction = from_tree_direction - to_tree_direction
                    if change_direction < 0:
                        tmp[a] = second_joint_node
                        tmp[joint_node] = a
                    else:
                        tmp[a] = joint_node

                    nextnode = tmp[a]
                    while nextnode != r:
                        succ_nodes.add(nextnode)
                        nextnode = tmp[nextnode]
                else:
                    tmp[a] = joint_node

                for s in S + A:
                    if s not in succ_nodes:
                        if oddist[s, a] < oddist[s, tmp[s]]:
                            tmp[s] = a
                sindexes = list(tmp.keys())
                tindexes = [tmp[s] for s in sindexes]
                cost = np.sum(oddist[sindexes, tindexes])
                if cost - last_cost < minCost:
                    minCost = cost - last_cost
                    minNode = a
                    minP = tmp
            # ! node may be 0, you have to write not None here
            if (
                minNode is not None
            ):  # * the best inc node that can reduced most cost of kth tree
                heappush(candiateIncNodes, (minCost, k, minNode, minP))
            else:
                continue
        # ! find the best and update the rest
        if (
            not candiateIncNodes
        ):  # ! this will happen if M is too big that no more tree needs
            break
        reducedCost, key_to_change, aggr_to_add, paths_to_replace = heappop(
            candiateIncNodes
        )
        # tree = construct_tree_from_paths(paths_to_replace, odpath)
        if Debug:
            print(f"choose key {key_to_change} set aggr {aggr_to_add}")
        # * update the chosen key's Paths
        for s, t in paths_to_replace.items():
            for n in odpath[s, t]:
                if n not in forbiddens and n not in As[key_to_change]:
                    potential_aggrs[key_to_change].add(
                        n
                    )  # ! find more potential aggregation nodes
        potential_aggrs[key_to_change].remove(aggr_to_add)
        Ps[key_to_change] = paths_to_replace
        As[key_to_change].append(aggr_to_add)
        last_costs[key_to_change] = last_costs[key_to_change] + reducedCost
        # T = construct_tree_from_paths(Ps[key_to_change] , odpath)

        deal_keys = [
            key_to_change
        ]  # ! only the changed key tree need to recalc it cost again
        # * update the resource use
        affect_keys = update_resource(
            aggr_to_add, capacity, candiateIncNodes, forbiddens
        )
        deal_keys = deal_keys + affect_keys
        M = M - 1
    cost = 0
    for k in range(K):
        P = Ps[k]
        # T = construct_tree_from_paths(P, odpath)
        sindexes = list(P.keys())
        tindexes = [P[s] for s in sindexes]
        c = np.sum(oddist[sindexes, tindexes])
        cost += c

    return cost, Ps, As


def test_mannual():
    random.seed(4441)
    a = "a"
    b = "b"
    c = "c"
    d = "d"
    e = "e"
    f = "f"
    g = "g"
    r = "r"
    h = "h"
    w = "w"
    G = nx.Graph(
        [
            (a, 4),
            (b, 4),
            (d, 7),
            (8, 2),
            (c, 1),
            (e, 2),
            (f, 8),
            (r, 3),
            (2, 4),
            (2, 6),
            (1, 4),
            (3, h),
            (5, 7),
            (1, 5),
            (5, 3),
            (3, 6),
            (4, 7),
            (6, 7),
            (g, 6),
            (1, 8),
            (8, 3),
        ]
    )
    nodes = {
        c: {"pos": (0, 1), "type": "host"},
        g: {"pos": (4, 2), "type": "host"},
        e: {"pos": (2, 1.5), "type": "host"},
        f: {"pos": (1, 0), "type": "host"},
        d: {"pos": (3, 3), "type": "host"},
        a: {"pos": (0, 2), "type": "host"},
        h: {"pos": (3, 0), "type": "host"},
        r: {"pos": (4, 1), "type": "host"},
        # w:{'pos':(2, 3),'type':'host'},
        1: {"pos": (1, 1), "type": "switch"},
        2: {"pos": (2, 1), "type": "switch"},
        4: {"pos": (1, 2), "type": "switch"},
        5: {"pos": (2, 2), "type": "switch"},
        b: {"pos": (1, 3), "type": "host"},
        6: {"pos": (3, 2), "type": "switch"},
        3: {"pos": (3, 1), "type": "switch"},
        7: {"pos": (2, 3), "type": "switch"},
        8: {"pos": (2, 0), "type": "switch"},
    }
    G.add_nodes_from([(node, attr) for node, attr in nodes.items()])
    switches = get_attr_nodes(G, "type", "switch")
    hosts = get_attr_nodes(G, "type", "host")
    G.add_nodes_from(hosts, capacity=0)
    for v in switches:
        G.nodes[v]["capacity"] = 1
    G_reindexed, nodeMap = get_reindexed_graph(G, len(hosts), len(switches))
    print(nodeMap)
    odpath, oddist = get_paths(G_reindexed)
    # plot(G)
    # plt.show()

    sources = [[nodeMap[n] for n in [a, b, c, d, e, g, f, h]]]
    targets = [nodeMap[r]]
    capacity = nx.get_node_attributes(G_reindexed, "capacity")
    cost, Ps, As = gstep(sources, targets, capacity, 1, oddist, odpath)
    g = construct_tree_from_paths(Ps[0], odpath)
    plot(g)
    plt.show()


def test_random():
    random.seed(2333)
    print("-" * 15 + "fattree" + "-" * 15)
    G = random_graph(15, 10).to_directed()
    hosts = get_attr_nodes(G, "type", "host")
    switches = get_attr_nodes(G, "type", "switch")
    # number_of_experiments = 10
    number_of_keys = 2
    M = [2]
    sources = []
    targets = []
    for _ in range(number_of_keys):
        load = 6
        ss = terminals = random.sample(hosts, load + 1)
        target = terminals.pop()
        print(f"sources: {ss}, dest: {target}")
        sources.append(ss)
        targets.append(target)
    set_capacity(G, 1)
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
        traffic = gstep(sources.copy(), targets.copy(), capacity.copy(), m, True)
        print(f"Algorith mat's result: {traffic}")
        print("-" * 40)


def test_fattree():
    random.seed(1415)
    G = fattree(4).to_directed()
    set_capacity(G, 20)
    capacity = nx.get_node_attributes(G, "capacity")
    print(capacity[32])
    sources = [[8, 3, 14, 6, 1, 9, 10, 15, 2, 4], [3, 13, 6, 14, 7, 1, 11, 10, 8, 12]]
    targets = [5, 2]

    traffic = mat(
        G.copy(), sources.copy(), targets.copy(), capacity.copy(), 4, Debug=True
    )


if __name__ == "__main__":
    test_mannual()
