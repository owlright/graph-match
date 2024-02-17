import matplotlib.pyplot as plt
from matplotlib import rcParams
import networkx as nx
import sys
import sys, os
import itertools


# ! I have to copy pairwise from utils.py to avoid circular import
def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


isDebug = True if sys.gettrace() else False

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
    "Consolas",
    "Tahoma",
    "DejaVu Sans",
    "Lucida Grande",
    "Verdana",
]


def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0,
):
    """Draw edge labels.
    Copied from https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5 * pos_1 + 0.5 * pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad * rotation_matrix @ d_pos
        ctrl_mid_1 = 0.5 * pos_1 + 0.5 * ctrl_1
        ctrl_mid_2 = 0.5 * pos_2 + 0.5 * ctrl_1
        bezier_mid = 0.5 * ctrl_mid_1 + 0.5 * ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def draw_aggr_graph(G, P, ax=None):
    ax = ax if ax else plt.gca()
    pos = nx.get_node_attributes(G, "pos")
    if pos == {}:
        pos = nx.nx_agraph.graphviz_layout(T, prog="dot")
    T = nx.DiGraph()
    for p in P:
        for node in p:
            T.add_node(node, type=G.nodes[node]["type"])
        for e in pairwise(p):
            T.add_edge(*e)
    plot(T, pos=pos)


def draw_flow_paths(G, P, ax=None):
    ax = ax if ax else plt.gca()
    pos = nx.get_node_attributes(G, "pos")
    if pos == {}:
        pos = nx.nx_agraph.graphviz_layout(T, prog="dot")
    T = nx.DiGraph()
    for i, p in enumerate(P):
        for e in pairwise(p):
            r_e = tuple(reversed(e))
            if T.has_edge(*e):
                T.edges[e]["rad"] += 0.1
            else:
                T.add_edge(e[0], e[1], rad=0)
                if T.has_edge(*r_e):
                    T.edges[e]["rad"] += 0.1
            e_rad = T.edges[e]["rad"]
            nx.draw_networkx_edges(
                T,
                pos,
                arrowstyle="->",
                arrowsize=10,
                width=1,
                edgelist=[e],
                edge_cmap=plt.get_cmap("tab10"),
                edge_vmin=0,
                edge_vmax=len(P),
                edge_color=[i],
                connectionstyle=f"arc3, rad = {e_rad}",
                ax=ax,
            )
    nodes_type = nx.get_node_attributes(G, "type")
    nx.draw_networkx_labels(
        G, pos, labels={n: n for n in G}, font_size=10, ax=ax  # resort to node's index
    )
    if nodes_type:
        hosts = [n for n, v in nodes_type.items() if v == "host"]
        switches = [n for n, v in nodes_type.items() if v == "switch"]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=hosts,
            node_shape="s",
            node_size=150,
            node_color="xkcd:white",
            edgecolors="k",
            linewidths=1,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=switches,
            node_shape="o",
            node_size=150,
            node_color="xkcd:white",
            edgecolors="k",
            linewidths=1,
            ax=ax,
        )


def plot(G: nx.DiGraph, node_label_name=None, edge_label_name=None, pos=None, ax=None):
    if not ax:
        ax = plt.gca()  # default is gca()

    if isDebug:
        ax.cla()  # * this is very helpful when you use plot() in debug, it will not draw on the same figure

    if not pos:
        pos = nx.get_node_attributes(G, "pos")
        if not pos:
            G.graph["rankdir"] = "BT"
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    nodes_type = nx.get_node_attributes(G, "type")
    node_color = nx.get_node_attributes(G, "color")
    # [G.nodes[n]['color'] for n in switches]
    if nodes_type:
        hosts = [n for n, v in nodes_type.items() if v == "host"]
        switches = [n for n, v in nodes_type.items() if v == "switch"]
        if node_color:
            switch_color = [node_color[n] for n in switches]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=hosts,
            node_shape="s",
            node_size=150,
            node_color="xkcd:white",
            edgecolors="k",
            linewidths=1,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=switches,
            node_shape="o",
            #  node_size=(9*mm*dpi)**2*0.5,
            node_size=250,
            node_color="xkcd:white" if not node_color else switch_color,
            edgecolors="k",
            linewidths=1,
            ax=ax,
        )
    else:
        nx.draw_networkx_nodes(
            G,
            pos,
            node_shape="s",
            node_size=150,
            node_color="xkcd:white",
            edgecolors="k",
            linewidths=1,
            ax=ax,
        )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=(
            nx.get_node_attributes(G, node_label_name)
            if node_label_name
            else {n: n for n in G}
        ),  # resort to node's index
        font_size=10,
        ax=ax,
    )
    all_edges = list(G.edges())
    edge_labels = nx.get_edge_attributes(G, edge_label_name)

    curved_edges = [e for e in all_edges if tuple(reversed(e)) in all_edges]
    straight_edges = list(set(all_edges) - set(curved_edges))
    if G.is_directed():
        arrow_style = "->"
        nx.draw_networkx_edges(
            G,
            pos,
            arrowstyle=arrow_style,
            arrowsize=10,
            width=1,
            edgelist=curved_edges,
            connectionstyle="arc3, rad = 0.1",
            ax=ax,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            arrowstyle=arrow_style,
            #  node_size = 150,
            arrowsize=10,
            width=1,
            edgelist=straight_edges,
            ax=ax,
        )
    else:
        assert curved_edges == []  # * undirected graph has no curved edges
        nx.draw_networkx_edges(
            G,
            pos,
            #  node_size = 150,
            arrowsize=10,
            width=1,
            edgelist=straight_edges,
            ax=ax,
        )

    edge_label_name = edge_label_name if edge_label_name else "weight"

    if edge_labels:
        curved_edge_labels = {edge: edge_labels[edge] for edge in curved_edges}
        straight_edge_labels = {edge: edge_labels[edge] for edge in straight_edges}
        my_draw_networkx_edge_labels(
            G, ax=ax, pos=pos, edge_labels=curved_edge_labels, rotate=False, rad=0.1
        )
        nx.draw_networkx_edge_labels(
            G,
            ax=ax,
            pos=pos,
            edge_labels=straight_edge_labels,
        )
    if isDebug:
        plt.show()


def draw_nodes(G, nodes, color_name="xkcd:white", ax=None):
    if not ax:
        ax = plt.gca()
    pos = nx.get_node_attributes(G, "pos")
    nx.draw_networkx_nodes(
        G,
        ax=ax,
        pos=pos,
        nodelist=nodes,
        node_size=100,
        node_color=color_name,
        edgecolors="k",
        linewidths=1,
    )


def draw_node_with_color(G, node_color, ax=None):
    if not ax:
        ax = plt.gca()
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        pos = nx.spring_layout(G)
    ncolors = [node_color[n] for n in G]
    nx.draw_networkx_nodes(
        G,
        ax=ax,
        pos=pos,
        node_size=150,
        node_color=ncolors,
        edgecolors="k",
        linewidths=1,
    )
    nx.draw_networkx_labels(G, ax=ax, pos=pos, font_size=10, labels={n: n for n in G})


def draw_edges(G, edge_attr=None, key=None, edge_color="r", ax=None):
    """draw edges with specified attr
    if also provided parameter key, it means find the key's value in attr
    """
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        pos = nx.spring_layout(G)
    if key != None:
        edges_dict = {
            (u, v): d[edge_attr][key]
            for u, v, d in G.edges(data=True)
            if key in d[edge_attr]
        }
    else:
        edges_dict = nx.get_edge_attributes(G, edge_attr)

    all_edges = G.edges()
    curved_edges = [edge for edge in edges_dict if tuple(reversed(edge)) in all_edges]
    straight_edges = list(set(edges_dict) - set(curved_edges))
    arrowsize = 10
    arrowwidth = 1
    arrowstyle = "->"
    nx.draw_networkx_edges(
        G,
        ax=ax,  # chosen edges are red
        pos=pos,
        edgelist=curved_edges,
        edge_color=edge_color,
        connectionstyle="arc3, rad = 0.13",
        arrowsize=arrowsize,
        arrowstyle=arrowstyle,
        width=arrowwidth,
    )
    nx.draw_networkx_edges(
        G,
        ax=ax,  # chosen edges are red
        pos=pos,
        edgelist=straight_edges,
        edge_color=edge_color,
        arrowsize=arrowsize,
        arrowstyle=arrowstyle,
        width=arrowwidth,
    )

    curved_edge_labels = {
        edge: round(edges_dict[edge] if edges_dict[edge] != {} else 0)
        for edge in curved_edges
    }
    straight_edge_labels = {
        edge: round(edges_dict[edge] if edges_dict[edge] != {} else 0)
        for edge in straight_edges
    }
    label_pos = 0.45
    bbox = dict(boxstyle="round", fc="w", ec="0.5", alpha=0)
    my_draw_networkx_edge_labels(
        G,
        ax=ax,
        font_size=7,
        pos=pos,
        edge_labels=curved_edge_labels,
        rotate=False,
        rad=0.15,
        bbox=bbox,
        label_pos=label_pos,
    )
    nx.draw_networkx_edge_labels(
        G,
        ax=ax,
        font_size=7,
        pos=pos,
        edge_labels=straight_edge_labels,
        bbox=bbox,
        label_pos=label_pos,
    )


def draw_labels(G, ax, node_attr, edge_attr, curved_edges, straight_edges):
    positions = nx.get_node_attributes(G, "pos")
    if node_attr:
        nx.draw_networkx_labels(
            G, ax=ax, labels=nx.get_node_attributes(G, node_attr), pos=positions
        )
    else:
        nx.draw_networkx_labels(G, ax=ax, pos=positions)
    if edge_attr:
        edge_labels = nx.get_edge_attributes(G, edge_attr)
        curved_edge_labels = {
            edge: edge_labels[edge] for edge in curved_edges if edge_labels[edge] > 0
        }
        straight_edge_labels = {
            edge: edge_labels[edge] for edge in straight_edges if edge_labels[edge] > 0
        }
        my_draw_networkx_edge_labels(
            G,
            ax=ax,
            pos=positions,
            edge_labels=curved_edge_labels,
            rotate=False,
            rad=0.15,
        )
        nx.draw_networkx_edge_labels(
            G,
            ax=ax,
            pos=positions,
            edge_labels=straight_edge_labels,
        )
    else:
        ...  # do nothing


def highlight(G, T, pos=None, color="tab:red", width=1) -> None:
    ax = plt.gca()
    if not pos:
        pos = nx.get_node_attributes(G, "pos")
        if not pos:
            pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(
        G,
        ax=ax,
        pos=pos,
        nodelist=T.nodes(),
        node_size=100,
        node_color=color,
        edgecolors="k",
        linewidths=1,
    )
    all_edges = list(G.edges())
    curved_edges = [e for e in T.edges() if reversed(e) in all_edges]
    straight_edges = list(set(T.edges()) - set(curved_edges))
    nx.draw_networkx_edges(
        T,
        pos,
        edge_color="r",
        arrowstyle="->",
        arrowsize=10,
        width=1,
        edgelist=curved_edges,
        connectionstyle="arc3, rad = 0.1",
    )
    nx.draw_networkx_edges(
        T,
        pos,
        edge_color="r",
        arrowstyle="->",
        arrowsize=10,
        width=1,
        edgelist=straight_edges,
    )
