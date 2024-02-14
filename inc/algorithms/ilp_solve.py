import networkx as nx
import random
import numpy as np
from heapq import heappop, heappush
import matplotlib.pyplot as plt
import itertools

import gurobipy as gp
from gurobipy import GRB  # pylint: disable=no-name-in-module
from gurobipy import quicksum  #  pylint: disable=no-name-in-module
# import logging

# logger = logging.getLogger(__name__)
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.getcwd())
    from inc.topo import minitopo
    from inc.util.utils import *
    from inc.algorithms.extract_paths import extract_paths
else:
    from ..util.utils import *
    # import coloredlogs
    # FORMAT = '[%(levelname)s: %(funcName)s %(lineno)3d] %(message)s'
    # coloredlogs.install(level='debug', logger=logger, fmt=FORMAT)
# import my own code
# try:
#     from ..util import *
# except ImportError:
#     import sys
#     import os
#     sys.path.append(os.getcwd())
#     from util import *
#     from topo import *


def binary_ilp_solve(sources, targets, S, arcs, capactity, M):
    # pylint: disable=no-member
    assert len(sources) == len(targets)
    keys = list(range(len(sources)))
    env = gp.Env(empty=True)
    # slience the license output

    env.setParam("OutputFlag", 1)
    env.start()
    model = gp.Model('mintotal_TrafficAggr', env=env)
    x = model.addVars(keys, arcs, vtype=GRB.INTEGER, lb=0, name='x')  # flow may be compressed
    f = model.addVars(keys, arcs, vtype=GRB.INTEGER, lb=0, name='f')  # flow cannot be compressed
    a = model.addVars(keys, S, vtype=GRB.BINARY, name="a")  # if allocate memory to key k in node i
    # if not set lb default is 0
    xloop = model.addVars(keys, S, vtype=GRB.CONTINUOUS, lb=0, name='xloop')
    model.addConstr((a.sum('*', '*') <= M), "job_memory")
    model.addConstrs((a.sum('*', i) <= capactity[i] for i in S), "switch_memory")

    for k, S0, target in zip(keys, sources, targets):
        Sf = [n for n in S if n not in S0 and n != target]
        # ! not allow source nodes to transmit the flow
        model.addConstrs((f.sum(k, i, '*') == 1 for i in S0), f"key_{k}_noncompressed_flow_must_out_at_S0")
        model.addConstrs((f.sum(k, i, '*') - f.sum(k, '*', i) == 0 for i in Sf),
                            f"key_{k}_noncompressed_flow_must_conserve_at_Sf")
        model.addConstr((-f.sum(k, '*', target) == -len(S0)), f"key_{k}_noncompressed_flow_must_end_at_target")

        model.addConstrs((x.sum(k, i, '*') - x.sum(k, '*', i) + xloop[k, i] == 0 for i in Sf),
                            f"key_{k}_flow_aggregate_with_xloop")
        model.addConstrs((x.sum(k, i, '*') - x.sum(k, '*', i) == 1 for i in S0),
                            f"key_{k}_flow_aggregate_without_xloop")

        model.addConstrs((xloop[k, i] <= len(S0) * a[k, i] for i in S), f"key_{k}_turnon_selfloop")
        model.addConstrs((xloop[k, i] >= a[k, i] for i in S), f"key_{k}_mustbeusedasBlue")

        model.addConstrs((len(S0) * x[k, i, j] >= f[k, i, j] for i, j in arcs), f"key_{k}_force_x_on_chosen_edge1")
        model.addConstrs((x[k, i, j] <= f[k, i, j] for i, j in arcs), f"key_{k}_force_x_on_chosen_edge2")
    model.addConstr(quicksum(x[k, i, j] for k in keys for i, j in arcs) <= 15, "upbound")
    model.setObjective(0, gp.GRB.MINIMIZE)
    model.optimize()


# ! this model deal with load(s) = 1
def ilp_solve_traffic(sources, targets, S, arcs, capactity, M, silence=False, reuse=True):
    '''use gurobi to solve the ilp problem
    Parameters
    ---------
    S0: sender hosts
    Sf: G.nodes except S0 and target(!note there are reduant hosts here)
    target: the receiver host
    arcs: G.edges
    context: any infomation you want to pass, here is 'load'
    M: total memory use
    '''
    # pylint: disable=no-member
    result = (None, None, None, None, None)
    assert len(sources) == len(targets)
    keys = list(range(len(sources)))

    if not reuse:
        env = gp.Env(empty=True)
        # slience the license output
        if silence:
            env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model('mintotal_TrafficAggr', env=env)
        # all_Sfs = set()
        # for S0, target in zip(sources, targets):
        #   Sf = [n for n in S if n not in S0 and n != target]
        #   Sf = S
        #   all_Sfs = all_Sfs | set(Sf)

        # Prepare the variables

        x = model.addVars(keys, arcs, vtype=GRB.INTEGER, lb=0, name='x')  # flow may be compressed
        f = model.addVars(keys, arcs, vtype=GRB.INTEGER, lb=0, name='f')  # flow cannot be compressed
        a = model.addVars(keys, S, vtype=GRB.BINARY, name="a")  # if allocate memory to key k in node i
        # if not set lb default is 0
        xloop = model.addVars(keys, S, vtype=GRB.CONTINUOUS, lb=0, name='xloop')
        model.addConstr((a.sum('*', '*') <= M), "job_memory")
        model.addConstrs((a.sum('*', i) <= capactity[i] for i in S), "switch_memory")

        for k, S0, target in zip(keys, sources, targets):
            Sf = [n for n in S if n not in S0 and n != target]
            # Prepare the constraints
            # non-compressed flow
            # ! allow source nodes to transmit the flow
            # model.addConstrs((f.sum(k, i, '*') - f.sum(k, '*', i) == 1 for i in S0),
            #                  f"key_{k}_noncompressed_flow_must_out_at_S0")
            # ! not allow source nodes to transmit the flow
            model.addConstrs((f.sum(k, i, '*') == 1 for i in S0), f"key_{k}_noncompressed_flow_must_out_at_S0")
            model.addConstrs((f.sum(k, i, '*') - f.sum(k, '*', i) == 0 for i in Sf),
                             f"key_{k}_noncompressed_flow_must_conserve_at_Sf")
            model.addConstr((-f.sum(k, '*', target) == -len(S0)), f"key_{k}_noncompressed_flow_must_end_at_target")

            # model.addConstrs((x.sum(k, i, '*') - x.sum(k, '*', i) == 1 for i in S0),
            #                  "flow_conservation_x_S0")
            # model.addConstrs((x.sum(k, i, '*') - x.sum(k, '*', i) + xloop[k, i] == 1
            #                   for i in S0), f"key_{k}_flow_aggregate_with_xloop")
            model.addConstrs((x.sum(k, i, '*') - x.sum(k, '*', i) + xloop[k, i] == 0 for i in Sf),
                             f"key_{k}_flow_aggregate_with_xloop")
            model.addConstrs((x.sum(k, i, '*') - x.sum(k, '*', i) == 1 for i in S0),
                             f"key_{k}_flow_aggregate_without_xloop")
            # model.addConstrs((x.sum(k, i, '*') - x.sum(k, '*', i) + xloop[k, i] == 0
            #                   for i in Sf), f"key_{k}_flow_aggregate_with_xloop")
            model.addConstrs((xloop[k, i] <= len(S0) * a[k, i] for i in S), f"key_{k}_turnon_selfloop")
            model.addConstrs((xloop[k, i] >= a[k, i] for i in S), f"key_{k}_mustbeusedasBlue")
            # model.addConstrs((xloop[k, i] + 1 ==0 for i in S0))
            # model.addConstrs((x.sum(k, i, '*')== 1 for i in S0))
            model.addConstrs((len(S0) * x[k, i, j] >= f[k, i, j] for i, j in arcs), f"key_{k}_force_x_on_chosen_edge1")
            model.addConstrs((x[k, i, j] <= f[k, i, j] for i, j in arcs), f"key_{k}_force_x_on_chosen_edge2")
        model.setObjective(quicksum(x[k, i, j] for k in keys for i, j in arcs), gp.GRB.MINIMIZE)
        model.optimize()

        model.write("mTA.lp")
        model.write("mTA.sol")
        # logger.info(f"model variants: {model.NumVars}")
        # logger.info(f"model constraints: {model.NumConstrs}")
        if model.Status == GRB.OPTIMAL or GRB.TIME_LIMIT or GRB.INTERRUPTED:
            result = (model.ObjVal, model.getAttr('x', a), model.getAttr('x', x), model.getAttr('x', xloop),
                      model.getAttr('x', f))
    if reuse:
        # * fast load the model
        # logger.info(f"{os.getcwd()} using file mTA.sol...")
        a = {}
        f = {}
        x = {}
        xloop = {}

        with open('mTA.sol', 'r') as fp:
            next(fp)
            line = fp.readline().split()
            objVal = int(line[-1])
            line = fp.readline()
            while line:
                line = line.split()
                value = line[1]
                var_name = line[0].split('[')[0]
                var_index = line[0].split('[')[1].strip('[]').split(',')
                k = int(var_index[0])
                i = int(var_index[1])
                if 'a' == var_name:
                    a[k, i] = int(value)
                if 'xloop' == var_name:
                    xloop[k, i] = int(value)
                if 'f' == var_name:
                    j = int(var_index[2])
                    f[k, i, j] = int(value)
                if 'x' == var_name:
                    j = int(var_index[2])
                    x[k, i, j] = int(value)
                line = fp.readline().strip()
        result = (objVal, a, x, xloop, f)
    return result


# ! this model can deal with load(s) > 1, but each source only has one path.
# todo it's easy to modify it to deal with multipath

@deprecated
def ilp_based_on_edge(S0, Sf, target, arcs, key_range, loads, capactitys, M, Theta, silence=True, reuse=False):
    '''use gurobi to solve the ilp problem
    Parameters
    ---------
    S0: sender hosts
    Sf: G.nodes except S0 and target(!note there are reduant hosts here)
    target: the receiver host
    arcs: G.edges
    key_range: number of keys
    context: any infomation you want to pass, here is 'load'
    M: total memory use
    Theta: a big number
    '''

    result = (None, None, None, None, None)
    if reuse:
        # logger.info(f"{os.getcwd()} using file mTA.sol...")
        a = {}
        f = {}
        x = {}
        xloop = {}

        with open('mTA.sol', 'r') as fp:
            next(fp)
            line = fp.readline().split()
            objVal = int(line[-1])
            line = fp.readline()
            while (line):
                line = line.split()
                value = line[1]
                var_name = line[0].split('[')[0]
                var_index = line[0].split('[')[1].strip('[]').split(',')
                if 'a' == var_name:
                    i = int(var_index[0])
                    k = int(var_index[1])
                    a[i, k] = int(value)
                if 'xloop' == var_name:
                    i = int(var_index[0])
                    k = int(var_index[1])
                    xloop[i, k] = int(value)
                if 'f' == var_name:
                    i = int(var_index[0])
                    j = int(var_index[1])
                    k = int(var_index[2])
                    f[i, j, k] = int(value)
                if 'x' == var_name:
                    i = int(var_index[0])
                    j = int(var_index[1])
                    k = int(var_index[2])
                    x[i, j, k] = int(value)
                line = fp.readline().strip()
        result = (objVal, a, x, xloop, f)
    if not reuse:
        env = gp.Env(empty=True)
        # slience the license output
        if silence:
            env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model('mintotal_TrafficAggr', env=env)
        # Prepare the variables
        a = model.addVars(Sf, key_range, vtype=GRB.BINARY, name="a")
        x = model.addVars(arcs, key_range, vtype=GRB.INTEGER, lb=0, name='x')  # flow may be compressed
        xloop = model.addVars(Sf, key_range, vtype=GRB.INTEGER, lb=0, name='xloop')
        # y = model.addVars(arcs, key_range, vtype=GRB.BINARY, name='y') # indicate which edge is used
        f = model.addVars(arcs, key_range, vtype=GRB.INTEGER, lb=0, name='f')  # flow cannot be compressed

        # Prepare the constraints
        # non-compressed flow
        model.addConstrs((a.sum(i, '*') <= capactitys[i] for i in Sf), "switch_memory")
        model.addConstr((quicksum(a[i, k] for i in Sf for k in key_range) <= M), "job_memory")
        # model.addConstrs(
        #   (f.sum(i, '*', k) - f.sum('*', i, k) == 0 for i in Sf for k in key_range), "Sf_nodes")
        # model.addConstrs(
        #   (f.sum(i, '*', k) - f.sum('*', i, k) == G.nodes[i]['load'] for i in S0 for k in key_range), "S0_nodesf")
        # ! so here I assume only one path each source
        model.addConstrs((f.sum(i, '*', k) - f.sum('*', i, k) == 1 for i in S0 for k in key_range),
                         "noncompressed_flow_conservation_f_S0_Sf")
        model.addConstrs((f.sum(i, '*', k) - f.sum('*', i, k) == 0 for i in Sf for k in key_range),
                         "noncompressed_flow_conservation_f_S0_Sf")
        model.addConstrs((f.sum(target, '*', k) - f.sum('*', target, k) == -len(S0) for k in key_range), "r_nodes")
        # model.addConstrs(
        #   (f[i,j,k] <= Theta*y[i,j,k] for i, j in arcs for k in key_range), "edge_idicator1"
        # )
        # model.addConstrs(
        #   (y[i,j,k] <= f[i,j,k] for i,j in arcs for k in key_range), "edge_idicator2"
        # )

        # compressed flow
        # model.addConstrs(
        #   (x.sum(i, '*', k) - x.sum('*', i, k) == G.nodes[i]['load'] for i in S0 for k in key_range), "S0_nodesx")
        model.addConstrs((x.sum(i, '*', k) - x.sum('*', i, k) == loads[i] for i in S0 for k in key_range),
                         "flow_conservation_x_S0")
        model.addConstrs((x.sum(i, '*', k) - x.sum('*', i, k) + xloop[i, k] == 0 for i in Sf for k in key_range),
                         "compressed_flow_conservation_x_Sf")
        model.addConstrs((xloop[i, k] <= Theta * a[i, k] for i in Sf for k in key_range), "turnon_selfloop")

        model.addConstrs((Theta * x[i, j, k] >= f[i, j, k] for i, j in arcs for k in key_range),
                         "force_x_on_chosen_edge1")
        model.addConstrs((x[i, j, k] <= Theta * f[i, j, k] for i, j in arcs for k in key_range),
                         "force_x_on_chosen_edge2")
        model.setObjective(quicksum(x[i, j, k] for i, j in arcs for k in key_range), gp.GRB.MINIMIZE)
        model.optimize()

        model.write("mTA.lp")
        model.write("mTA.sol")
        # logger.info(f"model variants: {model.NumVars}")
        # logger.info(f"model constraints: {model.NumConstrs}")
        if model.Status == GRB.OPTIMAL or GRB.TIME_LIMIT or GRB.INTERRUPTED:
            result = (model.ObjVal, model.getAttr('x', a), model.getAttr('x', x), model.getAttr('x', xloop),
                      model.getAttr('x', f))

    return result


def ilp_solve(G: nx.DiGraph,
              sources,
              targets,
              capacity,
              M,
              oddist,
              reuse=False,
              Debug=False,
              show_aggr_graph=False,
              show_path_graph=False):
    '''
    Parameters
    ---------
    G : the directed whole graph
    sources : sender hosts
    target : the receiver host
    M : the total memory can use
    reuse_ : convient when debugging

    Returns
    ---------
    total total_traffic
    computation_nodes
    each key's flow path
    '''

    K = len(sources)
    S = list(G)
    arcs = list(G.edges())

    total_traffic, a, x, xloop, f_all = ilp_solve_traffic(sources,
                                                         targets,
                                                         S,
                                                         arcs,
                                                         capacity,
                                                         M,
                                                         silence=not Debug,
                                                         reuse=reuse)

    key_compution_nodes = {k: [] for k in range(K)}
    Sf = [n for n in G if G.nodes[n]['type'] == 'switch']
    for k in range(K):
        for i in Sf:
            if a[k, i] > 0:
                key_compution_nodes[k].append(i)
    Ps = {
        # k: get_branches(G, k, sources[k], targets[k], x, key_compution_nodes[k])
        k: extract_paths(sources[k], [targets[k]] + key_compution_nodes[k], oddist)
        # k: extract_tree_paths(G, sources[k], targets[k], key_compution_nodes[k])
        for k in range(K)
    }

    cost = 0
    for k in range(K):
        P = Ps[k]
        sindexes = list(P.keys())
        tindexes = [P[s] for s in sindexes]
        c = np.sum(oddist[sindexes, tindexes])
        cost += c
    # logger.info(f"extracted traffic: {cost}")
    # ! The code below is to draw picture of aggr graph and key path
    if show_aggr_graph:
        positions = nx.get_node_attributes(G, 'pos')
        # S0_num_max = max([len(s) for s in sources])
        if show_path_graph:
            fig, axes = plt.subplots(K, 2)
            fig.canvas.manager.set_window_title("Aggregation Graph and Path for each Source")
        else:
            fig, axes = plt.subplots(K, 1)
            fig.canvas.manager.set_window_title("Aggregation Graph")

        if hasattr(axes, 'flat'):
            ax = iter(axes.flat)
        else:
            ax = iter([axes])
        for k in range(K):
            S0 = sources[k]
            target = targets[k]
            Sf = [n for n in S if n not in S0 and n != target]
            color_node, node_color = get_node_color(S0, Sf, target, key_compution_nodes[k])

            draw_key_aggr_graph(positions, arcs, node_color, k, x, next(ax))

            if show_path_graph:
                draw_branches(Ps[k], node_color, positions, next(ax))

        leftblank = 0.014
        bottomblank = 0.026
        plt.subplots_adjust(top=1 - bottomblank,
                            bottom=bottomblank,
                            right=1 - leftblank,
                            left=leftblank,
                            wspace=0,
                            hspace=0.057)
        plt.show()

    return int(total_traffic), Ps, key_compution_nodes

@deprecated
def get_branches(G, k, sources, target, x, aggr_nodes):
    branches = []
    arcs = G.edges()
    aggrs = aggr_nodes.copy()
    aggrs.append(target)
    S0 = sources
    aggr_inflows_num = {aggr: sum([x[k, i, j] for i, j in arcs if j == aggr]) for aggr in aggrs}
    aggr_inflows = {aggr: [] for aggr in aggrs}
    while aggr_inflows_num:
        for s in S0:
            paths = nx.single_source_shortest_path(G, s)
            d = G.number_of_edges()
            n = None
            # find the shortest node on the tree
            for aggr in aggr_inflows_num:
                if len(paths[aggr]) < d:
                    d = len(paths[aggr])
                    n = aggr
            aggr_inflows[n].append(s)
            aggr_inflows_num[n] -= 1
            branches.append(paths[n])
        del_aggrs = [k for k, d in aggr_inflows_num.items() if d == 0]
        for a in del_aggrs:
            del aggr_inflows_num[a]
        S0 = del_aggrs
    return branches

@deprecated
def extract_tree_paths(G, sources, target, aggr_nodes):
    T = [target]
    S = aggr_nodes + sources
    q = []
    visited = set()
    P = []
    # no use, just incase two paths have the same length,then compare path with letter in it.
    c = itertools.count()
    while S:
        for t in T:
            for s in S:
                if (s, t) not in visited:
                    visited.add((s, t))
                    length, path = nx.single_source_dijkstra(G, s, t)
                    heappush(q, (length, next(c), path))
        l, _, p = heappop(q)
        if p[0] not in S:
            continue
        P.append(p)
        S.remove(p[0])
        if p[0] in aggr_nodes:
            T.append(p[0])
    return P

def get_node_color(S0, Sf, target, key_compution_nodes):
    '''get aggr node for each key'''
    color_node = {}
    node_color = {}
    color_node['xkcd:blue'] = key_compution_nodes
    color_node['xkcd:yellow'] = [target]
    color_node['xkcd:grey'] = S0
    color_node['xkcd:white'] = list(set(S0 + Sf + [target]) -
                                    set(itertools.chain.from_iterable(color_node.values())))  # the rest nodes are white
    for color, nodes in color_node.items():
        for n in nodes:
            node_color[n] = color
    return color_node, node_color


def draw_branches(branches, node_color, pos, ax=None):
    ax = ax if ax else plt.gca()
    T = nx.DiGraph()
    nBranches = len(branches)
    for i, branch in enumerate(branches):
        for e in pairwise(branch):
            r_e = tuple(reversed(e))
            if T.has_edge(*e):
                T.edges[e]['rad'] += 0.1
            else:
                T.add_edge(e[0], e[1], rad=0)
                if T.has_edge(*r_e):
                    T.edges[e]['rad'] += 0.1
            e_rad = T.edges[e]['rad']
            nx.draw_networkx_edges(T,
                                   pos,
                                   arrowstyle='->',
                                   arrowsize=10,
                                   width=1,
                                   edgelist=[e],
                                   edge_cmap=plt.get_cmap("tab10"),
                                   edge_vmin=0,
                                   edge_vmax=nBranches,
                                   edge_color=[i],
                                   connectionstyle=f'arc3, rad = {e_rad}',
                                   ax=ax)

    ncolors = [node_color[n] for n in T]

    nx.draw_networkx_nodes(T,
                           pos,
                           node_shape="s",
                           node_size=150,
                           node_color=ncolors,
                           edgecolors='k',
                           linewidths=1,
                           ax=ax)


def draw_key_aggr_graph(pos, arcs, node_color, k, x, ax):
    ax = ax if ax else plt.gca()

    T = nx.DiGraph()
    T.add_nodes_from([(v, {'pos': p}) for v, p in pos.items()])
    for i, j in arcs:
        if x[k, i, j] > 0:
            T.add_edge(i, j, real_traffic=x[k, i, j])

    draw_node_with_color(T, node_color, ax)
    draw_edges(T, 'real_traffic', None, 'k', ax)

    ax.set_ylabel(f"key = {k}")


def test():
    random.seed(2333)
    G = random_graph(7, 5, 0.4)
    # G = jellyfish(7, 3, 2).to_directed()
    plot(G)
    hosts = get_attr_nodes(G, 'type', 'host')
    number_of_keys = 1
    sources = []
    targets = []
    for _ in range(number_of_keys):
        load = 4
        ss = terminals = random.sample(hosts, load + 1)
        target = terminals.pop()
        sources.append(ss)
        targets.append(target)
    set_capacity(G)

    ilp_solve(G, sources, targets, 2, False, False, True, True)


def test1():
    random.seed(4441)
    G = minitopo()
    hosts_num = len(get_attr_nodes(G, 'type', 'host'))
    switches_num = len(get_attr_nodes(G, 'type', 'switch'))
    G, node2index = get_reindexed_graph(G, hosts_num, switches_num)
    G = G.to_directed()

    odpath = dict(nx.all_pairs_dijkstra_path(G))
    oddist = np.empty([len(G), len(G)], dtype=int)
    for i in G:
        for j in G:
            oddist[i, j] = len(odpath[i][j]) - 1

    switches = get_attr_nodes(G, 'type', 'switch')
    hosts = get_attr_nodes(G, 'type', 'host')
    G.add_nodes_from(hosts, capacity=0)

    for v in switches:
        G.nodes[v]['capacity'] = 1
    G.nodes[4]['capacity'] = 0
    G.nodes[3]['capacity'] = 0
    # G.nodes[1]['capacity'] = 0
    # G.nodes[6]['capacity'] = 0
    sources = [[node2index[i] for i in ['a', 'b', 'c', 'd', 'e', 'g', 'f', 'h']]]
    print(f'{sources=}')
    targets = [node2index['r']]
    print(targets)
    capacity = nx.get_node_attributes(G, 'capacity')
    tc = ilp_solve(G, sources, targets, capacity, 3, oddist, False, True, True, False)
    print(tc)

def test_binary_solve():
    random.seed(4441)
    G = minitopo()
    hosts_num = len(get_attr_nodes(G, 'type', 'host'))
    switches_num = len(get_attr_nodes(G, 'type', 'switch'))
    G, node2index = get_reindexed_graph(G, hosts_num, switches_num)
    G = G.to_directed()

    odpath = dict(nx.all_pairs_dijkstra_path(G))
    oddist = np.empty([len(G), len(G)], dtype=int)
    for i in G:
        for j in G:
            oddist[i, j] = len(odpath[i][j]) - 1

    switches = get_attr_nodes(G, 'type', 'switch')
    hosts = get_attr_nodes(G, 'type', 'host')
    G.add_nodes_from(hosts, capacity=0)

    for v in switches:
        G.nodes[v]['capacity'] = 1
    G.nodes[4]['capacity'] = 0
    G.nodes[3]['capacity'] = 0
    # G.nodes[1]['capacity'] = 0
    # G.nodes[6]['capacity'] = 0
    sources = [[node2index[i] for i in ['a', 'b', 'c', 'd', 'e', 'g', 'f', 'h']]]
    print(f'{sources=}')
    targets = [node2index['r']]
    print(targets)
    capacity = nx.get_node_attributes(G, 'capacity')
    S = list(G)
    arcs = list(G.edges())
    binary_ilp_solve(sources, targets, S, arcs, capacity, 3)

if __name__ == "__main__":
    test_binary_solve()
