import numpy as np
from numba import njit
from numba import types
from numba.typed import List, Dict
from numba import boolean, int32, int64, float32, double

DTYPE = np.int32
DTYPE_MAX = np.iinfo(DTYPE).max


@njit("(i4[:])(i4[:], i4[:], i4[:, :])")
def extract_paths_njit_nodict(S, mstnodes, oddist):
    """Given aggregation tree's Paths, source nodes S, computation nodes A and receiver node r,
    return the new generated Paths according to Prim algo.

    Returns
    ----------
    P : the paths of tree S \rightarrow r through all nodes in A
    """
    # todo index array source array dest array
    # P = Dict.empty(
    #   key_type=np.int64,
    #   value_type=np.int64,
    # )

    # ! Step 1. extract the paths form computation nodes to target
    # mstNodes = np.concatenate(List([r]) ,A)
    r = mstnodes[0]  # ! mstnodes[0] must be target
    Slen = len(S)
    Tlen = len(mstnodes)  # number of nodes in the mstT
    # TSeq = np.arange(Tlen, dtype=np.int32)

    # * similar to Prim algo
    in_mst_set = np.zeros(Tlen, dtype=DTYPE)
    P = np.full(Tlen + Slen, -1, dtype=DTYPE)
    parent = np.empty(Tlen, dtype=DTYPE)
    dist = np.full(Tlen, DTYPE_MAX, dtype=DTYPE)
    dist[0] = 0
    parent[0] = -1
    for _ in range(Tlen):
        # * get the closet node to the tree
        d = DTYPE_MAX
        for i in range(Tlen):
            if in_mst_set[i] == 0 and dist[i] < d:
                d = dist[i]
                closest_index = i
                closest_node = mstnodes[i]
        in_mst_set[closest_index] = 1  # insert this node into tree

        # * connect the closest path to T
        if closest_node != r:
            P[closest_index] = mstnodes[parent[closest_index]]
        # * update the rest nodes' distance to T
        for i in range(Tlen):
            v = mstnodes[i]
            if in_mst_set[i] == 0 and oddist[v, closest_node] < dist[i]:
                dist[i] = oddist[v, closest_node]
                parent[i] = closest_index
    # ! Step 2. adjust the paths from sources to computation nodes or target
    for i in range(Slen):
        s = S[i]
        d = DTYPE_MAX
        for j in range(Tlen):
            if oddist[s, mstnodes[j]] < d:
                d = oddist[s, mstnodes[j]]
                P[Tlen + i] = mstnodes[j]

    return P


@njit("DictType(i8, i8)(i4[:], i4[:], i4[:, :])")
def extract_paths_njit(S, mstnodes, oddist):
    """Given aggregation tree's Paths, source nodes S, computation nodes A and receiver node r,
    return the new generated Paths according to Prim algo.

    Returns
    ----------
    P : the paths of tree S \rightarrow r through all nodes in A
    """
    P = Dict.empty(
        key_type=np.int64,  # see https://github.com/numba/numba/issues/8676
        value_type=np.int64,
    )

    # ! Step 1. extract the paths form computation nodes to target
    # mstNodes = np.concatenate(List([r]) ,A)
    r = mstnodes[0]  # ! mstnodes[0] must be target
    Tlen = len(mstnodes)  # number of nodes in the mstT
    # TSeq = np.arange(Tlen, dtype=np.int32)

    # * similar to Prim algo
    in_mst_set = np.zeros(Tlen, dtype=DTYPE)
    parent = np.empty(Tlen, dtype=DTYPE)
    dist = np.full(Tlen, DTYPE_MAX, dtype=DTYPE)
    dist[0] = 0
    parent[0] = -1
    for _ in range(Tlen):
        # * get the closet node to the tree
        d = DTYPE_MAX
        for i in range(Tlen):
            if in_mst_set[i] == 0:
                if dist[i] < d:
                    d = dist[i]
                    closest_index = i
                    closest_node = mstnodes[i]
        in_mst_set[closest_index] = 1  # insert this node into tree
        # uind = np.argmin(dist[in_mst_nodes])
        # uT = TSeq[in_mst_nodes][uind]
        # u = mstnodes[uT] # ! the current closest node is u
        # in_mst_nodes[uT] = False

        # * connect the closest path to T
        if closest_node != r:
            P[closest_node] = mstnodes[parent[closest_index]]
        # * update the rest nodes' distance to T
        for i in range(Tlen):
            v = mstnodes[i]
            if in_mst_set[i] == 0 and oddist[v, closest_node] < dist[i]:
                dist[i] = oddist[v, closest_node]
                parent[i] = closest_index
    # ! Step 2. extract the paths from sources to computation nodes or target
    for i in range(len(S)):
        s = S[i]
        d = DTYPE_MAX
        for j in range(Tlen):
            if oddist[s, mstnodes[j]] < d:
                d = oddist[s, mstnodes[j]]
                P[s] = mstnodes[j]

    return P


def extract_paths(S: dict, mstnodes: list, oddist: np.ndarray) -> dict:
    """Given aggregation tree's Paths, source nodes S, computation nodes A and receiver node r,
    return the new generated Paths according to Prim algo.

    Returns
    ----------
    P : the paths of tree S \rightarrow r through all nodes in A
    """
    P = {}
    # ! Step 1. extract the paths form computation nodes to target
    # mstNodes = np.concatenate(List([r]) ,A)
    r = mstnodes[0]  # ! mstnodes[0] must be target
    Tlen = len(mstnodes)  # number of nodes in the mstT
    # TSeq = np.arange(Tlen, dtype=np.int32)

    # * similar to Prim algo
    in_mst_set = np.zeros(Tlen, dtype=DTYPE)
    parent = np.empty(Tlen, dtype=DTYPE)
    dist = np.full(Tlen, np.iinfo(DTYPE).max, dtype=DTYPE)
    dist[0] = 0
    parent[0] = -1
    for _ in range(Tlen):
        # * get the closet node to the tree
        d = np.iinfo(DTYPE).max
        for i in range(Tlen):
            if in_mst_set[i] == 0:
                if dist[i] < d:
                    d = dist[i]
                    closest_index = i
                    closest_node = mstnodes[i]
        in_mst_set[closest_index] = 1  # insert this node into tree
        # uind = np.argmin(dist[in_mst_nodes])
        # uT = TSeq[in_mst_nodes][uind]
        # u = mstnodes[uT] # ! the current closest node is u
        # in_mst_nodes[uT] = False

        # * connect the closest path to T
        if closest_node != r:
            P[closest_node] = mstnodes[parent[closest_index]]
        # * update the rest nodes' distance to T
        for i in range(Tlen):
            v = mstnodes[i]
            if in_mst_set[i] == 0 and oddist[v, closest_node] < dist[i]:
                dist[i] = oddist[v, closest_node]
                parent[i] = closest_index
    # ! Step 2. extract the paths from sources to computation nodes or target
    for i in range(len(S)):
        s = S[i]
        d = np.iinfo(DTYPE).max
        for j in range(Tlen):
            if oddist[s, mstnodes[j]] < d:
                d = oddist[s, mstnodes[j]]
                P[s] = mstnodes[j]

    return P
