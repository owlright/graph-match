import networkx as nx
import matplotlib.pyplot as plt
import itertools

# * turn node posion to number
xyz2str = lambda x, y, z: f"{x}{y}{z}"
toten = lambda origin, n: int(origin, base=n)
ten2xyz = lambda origin, n: (origin // n**2, origin % n**2 // n, origin % n)


def torus(n=3) -> nx.Graph:
    G = nx.empty_graph()
    for x, y, z in itertools.product(range(n), repeat=3):
        G.add_node((x, y, z), type="switch", pos=(x, y, z))
        G.add_node(x * n**2 + y * n + z, type="host")
        G.add_edge((x, y, z), x * n**2 + y * n + z)

    for x, y, z in itertools.product(range(n), repeat=3):
        up = x, y, (z + 1) % n
        down = x, y, (z - 1) % n
        left = (x - 1) % n, y, z
        right = (x + 1) % n, y, z
        front = x, (y - 1) % n, z
        behind = x, (y + 1) % n, z
        G.add_edges_from(
            zip(itertools.repeat((x, y, z), 6), [up, down, left, right, front, behind])
        )
    return G


if __name__ == "__main__":
    G = torus(2)
    nx.draw(G, with_labels=True)
    plt.show()
