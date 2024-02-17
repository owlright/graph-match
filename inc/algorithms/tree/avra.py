import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

from itertools import pairwise

try:
    from ...util.utils import *
except:
    import sys, os

    sys.path.append(os.getcwd())
    from inc.util.utils import *
    from inc.topo.fattree import fattree
    import random
    from rich import print as rprint


def avra(G: nx.Graph, sources: list, root: int, Debug=False) -> nx.DiGraph:
    assert G.graph["name"] == "fattree"
    if len(sources) < 1:
        assert False
    senders = set(sources)
    fir = senders.pop()
    # sources.remove(fir)
    if Debug:
        rprint(f"left sources to add: {sources}")
        rprint(f"find shortest path ({fir}, {root})")

    first_path = nx.shortest_path(G, fir, root)
    T = nx.DiGraph()
    for u, v in pairwise(first_path):
        T.add_edge(u, v)

    node_visited = {n: False for n in G}
    # G.add_nodes_from(G.nodes(), visited=False)

    while sources:
        # logger.debug(f"add {sources[-1]}...")
        T = hook(T, G, node_visited.copy(), sources.pop())
        # logger.debug("success.")
        # G.add_nodes_from(G.nodes(), visited=False)
    assert nx.is_tree(T) and max(d for n, d in T.out_degree()) <= 1
    update_graph(T, G)
    return T


def hook(tree: nx.DiGraph, G: nx.Graph, visited: dict, member: int):
    # tree = get_attr_nodes(G, 'intree', True)
    # tree = T.nodes()
    neighbors = G.neighbors
    level = lambda n: G.nodes[n]["level"]

    def rand_upper_neigh(m: int) -> int:
        """ramdom pick up a upper layer node"""
        this_level = level(m)
        return random.choice([n for n in neighbors(m) if level(n) > this_level])

    for adj in neighbors(member):
        if adj in tree:
            tree.add_edge(member, adj)
            # logger.debug(f"add edge ({member},{adj})")
            return tree
        visited[adj] = True

    for adj in neighbors(member):
        for adj2 in neighbors(adj):
            if not visited[adj2]:
                if adj2 in tree:
                    tree.add_edge(member, adj)
                    tree.add_edge(adj, adj2)
                    return tree
                # logger.debug(f"add edge ({member},{adj},{adj2})")
                visited[adj2] = True

    if level(member) < 3:
        candiate = rand_upper_neigh(member)
        # G.add_nodes_from(G.nodes(), visited=False)
        # logger.debug(f"try to use candiate node {candiate}")
        tree_ = hook(tree, G, visited, candiate)
        if tree_ is not None:
            assert tree_ is tree
            tree.add_edge(member, candiate)
    else:
        return None

    # ! if program runs here which means
    # ! level(member) < 3
    if level(member) != 0:
        return None
    # if go up to the top level still cannot find a node
    # to join the tree, resort to bfs_path
    if tree_ is None:
        e = None, None
        while e[0] != member:
            e = bfs(G, tree, member)  # bfs search the first edge to the tree
            tree.add_edge(*e)

    return tree


# def bfs_node_to_tree(G:nx.Graph, tree, n):
#   q = deque([[n]])
#   visited = set([n])
#   while q:
#     path = q.popleft()
#     last_node = path[-1]

#     for n in G.neighbors(last_node):
#       if n in tree:
#         return path

#     for v in G.neighbors(last_node):
#       if v not in visited:
#         visited.add(v)
#         new_path = path.copy()
#         new_path.append(v)
#         q.append(new_path)


def bfs(G: nx.Graph, tree: nx.DiGraph, source):
    neighbors = lambda node: iter(G.neighbors(node))
    visited = {source}
    queue = deque([(source, neighbors(source))])  # ! source must be number
    while queue:
        parent, children = queue[0]
        for child in children:
            if child not in visited:
                visited.add(child)
                if tree.has_node(child):
                    return parent, child
                else:  # * if find one just return to up level
                    queue.append((child, neighbors(child)))
        queue.popleft()


def test_ava():
    G = fattree(4, False)
    terminals = [0, 1, 2, 4, 5]
    root = 3
    random.seed(234)
    T = avra(G, terminals.copy(), root, True)
    plot(T, "level")
    plt.show()


def test_eva_bfs():
    G = fattree(4, False)
    G.remove_edge(16, 25)
    G.remove_edge(17, 25)
    terminals = [0, 1, 2, 4, 5]
    root = 3
    random.seed(234)
    T = avra(G, terminals.copy(), root, True)
    plot(T)
    plt.show()


if __name__ == "__main__":

    test_eva_bfs()

    # nx.draw(H, ax=ax, pos=nx.get_node_attributes(T, 'pos'), with_labels=True)
    # highlight(G, ax, "tab:red",4, get_attr_nodes(T, 'intree', True), get_attr_edges(T, 'intree', True))
