import numpy as np
import networkx as nx
from .tree.construct import construct_tree_from_paths
from heapq import heappop, heappush, heapify


def greedy_tree(
    sources,
    targets,
    trees: list[nx.DiGraph],
    capacity: dict,
    M: int,
    oddist: np.ndarray,
    odpath,
) -> tuple[int, dict, list]:
    K = len(sources)
    As = [[] for _ in targets]

    Ps = {}
    flow_paths = {}
    # * remove all servers and exhausted switches
    forbiddens = set([n for n, v in capacity.items() if v == 0])
    for k in range(K):
        S = sources[k]
        r = targets[k]
        T = trees[k]
        T_succ = T._succ
        Ps[k] = {}
        flow_paths[k] = {}
        for s in S:
            Ps[k][s] = r
            n = s
            path = []
            while n != r:
                path.append(n)
                n = list(T_succ[n])[0]
            path.append(r)
            flow_paths[k][s] = path
            # forbiddens.add(s)
    candiate_aggnodes = []
    tree_keys = list(range(K))
    while M > 0:
        for k in tree_keys:
            inflows = {}
            dests = {}
            flow_path = flow_paths[k]
            A = As[k]
            # * count each node's in-flows number
            for s, p in flow_path.items():
                assert s == p[0]
                for node in p:
                    if node not in forbiddens and node not in A:
                        inflows[node] = inflows.get(node, 0) + 1
                        dests[node] = p[-1]
            bestNode = None
            bestCost = 0
            # * find nodes that have two in-flows
            for node, flows_num in inflows.items():
                if flows_num >= 2:
                    cost_reduced = flows_num * oddist[node, dests[node]]
                    # cost_can_reduced = flows_num * (len(flow_path[node]) - 1)
                    if cost_reduced > bestCost:
                        bestCost = cost_reduced
                        bestNode = node
            if bestNode is not None:
                heappush(candiate_aggnodes, (-bestCost, k, bestNode))
        if not candiate_aggnodes:
            # ! this will happen if M is too big that no more tree needs
            break
        _, key_to_change, aggr_to_add = heappop(candiate_aggnodes)
        P = Ps[key_to_change]
        flow_path = flow_paths[key_to_change]
        As[key_to_change].append(aggr_to_add)  # * record the aggrnode
        affect_flows = []

        # * find which flows will be affected by aggregation
        for s, p in flow_path.items():
            assert s == p[0]
            if aggr_to_add in p:  # merge the path
                t = p[-1]
                P[s] = aggr_to_add
                P[aggr_to_add] = t
                index = p.index(aggr_to_add)
                affect_flows.append((s, index))  # ! cannot change flow_path here

        # * aggregate the flows
        for s, index in affect_flows:
            flow_path[aggr_to_add] = flow_path[s][index:]
            flow_path[s] = flow_path[s][: index + 1]

        tree_keys = [key_to_change]  # ! only the changed key tree need to recalc cost
        capacity[aggr_to_add] -= 1
        replace_nodes = []
        # ! remove exhausted switch nodes
        if capacity[aggr_to_add] == 0:
            forbiddens.add(aggr_to_add)  # this node cannot be used anymore
            for item in candiate_aggnodes:
                if item[2] in forbiddens:  # those affected trees need to be recalced
                    tree_keys.append(item[1])
                else:
                    replace_nodes.append(item)
            candiate_aggnodes[:] = replace_nodes  # * just keep not affected trees
            heapify(candiate_aggnodes)
        M = M - 1

    # ! adjustment
    # * every flow should choose the shortest path to destionation
    for k in range(K):
        P = Ps[k]
        A = As[k]
        S = sources[k]
        r = targets[k]
        for s, t in P.items():
            if P[s] not in A and P[s] != r:
                for a in A:
                    P[s] = a if oddist[s, a] < oddist[s, P[s]] else r
    cost = 0

    for k in range(K):
        P = Ps[k]
        T = construct_tree_from_paths(P, odpath)
        # nouse = []
        # for n in As[k]:
        #   if T.in_degree(n) == 1:
        #     nouse.append(n)
        # for n in nouse:
        # As[k].remove(n)
        sindexes = list(P.keys())
        tindexes = [P[s] for s in sindexes]
        c = np.sum(oddist[sindexes, tindexes])
        cost += c

    return cost, Ps, As
