from heapq import heappop, heappush, heapify
import numpy as np


def gspt(sources, targets, capacity, M, oddist, odpath):
    K = len(sources)
    As = [[] for r in targets]

    Ps = {}
    flow_paths = {}
    forbiddens = set([n for n, v in capacity.items() if v == 0])
    for k in range(K):
        S = sources[k]
        r = targets[k]
        Ps[k] = {}
        flow_paths[k] = {}
        for s in S:
            Ps[k][s] = r
            flow_paths[k][s] = odpath[s, r]
            forbiddens.add(s)
        # T = construct_tree_from_paths(Ps[k], odpath)
        forbiddens.add(r)

    candiateIncNodes = []
    deal_keys = list(range(K))
    while M != 0:
        for k in deal_keys:
            inflows = {}
            dests = {}
            flow_path = flow_paths[k]
            A = As[k]
            # find aggregation nodes
            for _, p in flow_path.items():
                for node in p:
                    if node not in forbiddens and node not in A:
                        inflows[node] = inflows.get(node, 0) + 1
                        dests[node] = p[-1]
            bestNode = None
            bestCost = 0
            for node, flows_num in inflows.items():
                if flows_num >= 2:
                    cost_can_reduced = flows_num * oddist[node, dests[node]]
                    if cost_can_reduced > bestCost:
                        bestCost = cost_can_reduced
                        bestNode = node
            if bestNode is not None:
                heappush(candiateIncNodes, (-bestCost, k, bestNode))

        if (
            not candiateIncNodes
        ):  # ! this will happen if M is too big that no more tree needs
            break
        _, key_to_change, aggr_to_add = heappop(candiateIncNodes)

        P = Ps[key_to_change]
        flow_path = flow_paths[key_to_change]
        As[key_to_change].append(aggr_to_add)
        affect_flows = []
        # other_flows = []
        for _, p in flow_path.items():
            s = p[0]
            if aggr_to_add in p:  # merge the path
                t = p[-1]
                P[s] = aggr_to_add
                P[aggr_to_add] = t
                index = p.index(aggr_to_add)
                affect_flows.append((s, index))  # cannot change flow_path here
            # else:
            #   other_flows.append(s)
        for s, index in affect_flows:
            flow_path[aggr_to_add] = flow_path[s][index:]
            flow_path[s] = flow_path[s][: index + 1]
        # flow_path[aggr_to_add] = odpath[aggr_to_add, P[aggr_to_add]]
        # succ_nodes = set()
        # nextnode = P[aggr_to_add]
        # r = targets[key_to_change]
        # while nextnode != r:
        #   succ_nodes.add(nextnode)
        #   nextnode = P[nextnode]
        # for s in affect_flows:
        #   flow_path[s] = odpath[s, aggr_to_add]
        # for s in other_flows:
        #   if s not in succ_nodes:
        #     if oddist[s, aggr_to_add] < oddist[s, P[s]]:
        #       P[s] = aggr_to_add
        #       flow_path[s] = odpath[s, aggr_to_add]
        # T = construct_tree_from_paths(flow_path)
        deal_keys = [
            key_to_change
        ]  # ! only the changed key tree need to recalc it cost again
        capacity[aggr_to_add] -= 1
        candiateIncNodes_replace = []
        if capacity[aggr_to_add] == 0:
            forbiddens.add(aggr_to_add)  # this node cannot be used anymore
            for item in candiateIncNodes:
                if item[2] in forbiddens:
                    deal_keys.append(item[1])
                else:
                    candiateIncNodes_replace.append(item)
            candiateIncNodes[:] = candiateIncNodes_replace
            heapify(candiateIncNodes)
        M = M - 1

    for k in range(K):
        P = Ps[k]
        A = As[k]
        S = sources[k]
        r = targets[k]
        for s in S:  # ! we dont change the aggr flows direction
            # ! but the flow from sources should still be correct
            if P[s] == r:
                for a in A:
                    if oddist[s, a] < oddist[s, P[s]]:
                        P[s] = a
    cost = 0
    for k in range(K):
        P = Ps[k]
        # T = construct_tree_from_paths(P, odpath)
        sindexes = list(P.keys())
        tindexes = [P[s] for s in sindexes]
        c = np.sum(oddist[sindexes, tindexes])
        cost += c
    return cost, Ps, As
