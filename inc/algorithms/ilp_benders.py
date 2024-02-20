import numpy as np
from gurobipy import Model, GRB, quicksum
import random
import networkx as nx


def debug_model(m: Model):
    if m.Status == GRB.INFEASIBLE:
        print("The model is infeasible")
        m.computeIIS()
        m.write("debug_why_infeasible.ilp")
    else:
        m.write("debug_success.lp")
        m.write("debug_success.sol")
        for var in m.getVars():
            if var.X > 0:
                print(f"{var.varName}: {var.X}")
        print(f"最优值: {m.ObjVal}")


def example_topo() -> nx.Graph:
    random.seed(2321)  # ! this is important
    g = nx.random_graphs.random_regular_graph(4, 11)
    g.remove_edge(0, 8)
    g.remove_edge(2, 3)
    return g


def check_feasible(model: Model):
    m = model.copy()
    m.Params.OutputFlag = 0
    x = [(0, 10), (1, 10), (2, 5), (10, 5), (5, 4), (3, 4)]
    xnames = [f"x[{e[0]},{e[1]}]" for e in x]
    for t in m.getVars():
        if t.varName in xnames:
            m.addConstr(t == 1)
        elif t.varName[0] == "x":
            m.addConstr(t == 0)
    m.addConstr(m.getVarByName("a[10]") == 1)
    m.addConstr(m.getVarByName("a[5]") == 1)
    m.optimize()
    debug_model(m)


if __name__ == "__main__":
    import sys, os

    sys.path.append(os.getcwd())
    from inc.algorithms import *
    import matplotlib.pyplot as plt

    # ! 物理拓扑
    g = example_topo()
    g = g.to_directed()
    arcs = g.edges()
    # ! 设置权重、容量、成本
    capacity = {e: random.randint(10, 50) for e in arcs}
    node_cost = {e: random.randint(3, 12) for e in g.nodes()}
    edge_cost = {e: 1 for e in arcs}  # random.randint(10, 30)
    # 可视化
    pos = nx.kamada_kawai_layout(g)
    nx.set_node_attributes(g, node_cost, "cost")
    nx.set_node_attributes(g, pos, "pos")
    nx.set_edge_attributes(g, capacity, "cost")
    # nx.draw(g, pos, with_labels=True)
    # plt.savefig("graph.png")
    # ! 准备发送端和接收端
    S = [i for i in range(4)]
    r = 4

    m = Model("OriginalProblem")
    # ! 用y来标记经过的边
    x = m.addVars(arcs, vtype=GRB.INTEGER, name="x")  # 边上实际经过的流量
    y = m.addVars(
        arcs, vtype=GRB.INTEGER, name="y"
    )  # 如果汇聚点不起作用，边上会经过的流量

    m.addConstrs((y.sum(i, "*") - y.sum("*", i) == 1 for i in S), "FlowConservation0")

    m.addConstrs(
        (y.sum(i, "*") - y.sum("*", i) == 0 for i in set(g.nodes()) - set(S + [r])),
        "FlowConservation1",
    )
    m.addConstr(y.sum(r, "*") - y.sum("*", r) == -len(S), "FlowConservation2")

    m.addConstrs((x[e] <= y[e] for e in arcs), "MarkUsedEdges1")
    m.addConstrs((len(S) * x[e] >= y[e] for e in arcs), "MarkUsedEdges2")

    # ! 如果a[i] = 1, 则z[i] > 0，x就可以比y小
    a = m.addVars(g.nodes(), vtype=GRB.BINARY, name="a")
    m.addConstr(a[r] == 0, "DoNotUseTheRootAsAggregation")
    z = m.addVars(g.nodes(), vtype=GRB.CONTINUOUS, name="z")
    m.addConstrs(
        (
            x.sum(v, "*") - x.sum("*", v) + z[v] == 0
            for v in set(g.nodes()) - set(S + [r])
        ),
        "NonTerminal",
    )
    m.addConstrs(
        (x.sum(v, "*") - x.sum("*", v) + z[v] == 1 for v in S),
        "Terminal",
    )
    m.addConstrs(
        (z[i] <= len(S) * a[i] for i in set(g.nodes()) - {r}),
        "AllowAbsorb",
    )
    m.addConstrs(
        (z[i] >= a[i] for i in set(g.nodes()) - {r}),
        "AggregationNodeMustBeUsed",
    )
    m.addConstr(a.sum("*") <= 1, "Resource")
    m.setObjective(
        quicksum(edge_cost[e] * x[e] for e in arcs),
        GRB.MINIMIZE,
    )  # + quicksum(node_cost[v] * a[v] for v in g.nodes())

    m.update()
    # check_feasible(m)

    m.optimize()
    debug_model(m)
