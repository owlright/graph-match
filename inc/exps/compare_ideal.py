# * this file is for testing the performance of optim
try:
    from .bruteforce_exp import bruteforce_exp
    from .exp_setup import *
    from ..topo import *
    from ..algorithms import *
    from ..util.utils import *
    import time
except ImportError:
    import sys, os

    sys.path.append(os.getcwd())
    from inc.exps import *
    from inc.util import *
    from inc.topo import *


def compare_ideal(
    result_file_name, topo, number_of_experiments, Debug=False, silent=True
):
    result_file = open(result_file_name, "w")
    result_file.write(str(number_of_experiments) + "\n")
    if not silent:
        print("=" * 26 + topo + "=" * 26)
    result_file.write(topo + "\n")
    number_of_sources = 11
    tree_algos = ["OPT", "Shortest", "Takashami", "GCAT"]
    if topo == "fattree":
        G = fattree(8)
        tree_algos.append("Avalanche")
    elif topo == "torus":
        nodes_each_edge = 8
        G = torus(nodes_each_edge)
        tree_algos.append("Camdoop")
        number_of_sources = 10
    elif topo == "bcube":
        network_level = 3
        G = bcube(network_level, 4)
        tree_algos.append("IRS")
        number_of_sources = 10
    elif topo == "random":
        N = 400
        G = random_graph(N)
    G, _ = get_reindexed_graph(G)
    hosts = get_attr_nodes(G, "type", "host")
    switches = get_attr_nodes(G, "type", "switch")
    odpath, oddist = get_paths(G)
    result_file.write(f"{tree_algos}\n")

    for expno in range(number_of_experiments):
        start = time.perf_counter()
        if not silent:
            print(f"Current Expno: {expno}")
        # result_file.write(str(expno)+"\n")
        if topo == "random":
            G = random_graph(N)
            G, _ = get_reindexed_graph(G)
            odpath, oddist = get_paths(G)
            hosts = get_attr_nodes(G, "type", "host")  # ! must do this again
            switches = get_attr_nodes(G, "type", "switch")

        S = random.sample(hosts, number_of_sources + 1)
        r = S.pop()
        set_capacity(G, 0)

        chosen_switches = random.sample(switches, int(100))
        set_capacity(G, 1, chosen_switches)
        capacity = nx.get_node_attributes(G, "capacity")

        M = number_of_sources - 1
        # todo cost and enough arrray
        last_costs = [np.sum(oddist[S, r])] * len(tree_algos)
        last_used = [0] * len(tree_algos)

        result_file.write(f"{last_costs[0]}\n")
        enoughs = [False] * len(tree_algos)

        for m in range(1, M):
            if all(enoughs):
                break
            solution = []
            for index, alg in enumerate(tree_algos):
                if not enoughs[index]:
                    if alg == "OPT":
                        cost, P, A = bruteforce_exp(S, r, capacity.copy(), m, oddist)
                    elif alg == "Shortest":
                        shortest_tree = build_sptree(S, r, odpath)
                        cost, Ps, As = greedy_tree(
                            [S],
                            [r],
                            [shortest_tree],
                            capacity.copy(),
                            m,
                            oddist,
                            odpath,
                        )
                        P = Ps[0]
                        A = As[0]
                    elif alg == "Takashami":
                        if topo != "bcube":
                            steiner_tree = build_steiner_tree(
                                S, r, oddist, odpath, set(hosts)
                            )
                        else:
                            steiner_tree = build_steiner_tree(S, r, oddist, odpath)
                        cost, Ps, As = greedy_tree(
                            [S], [r], [steiner_tree], capacity.copy(), m, oddist, odpath
                        )
                        P = Ps[0]
                        A = As[0]
                    elif alg == "GCAT":
                        cost, Ps, As = gstep(
                            [S], [r], capacity.copy(), m, oddist, odpath
                        )
                        P = Ps[0]
                        A = As[0]
                    elif alg == "IRS":
                        tree = irs_based(G, S, r, network_level)
                        cost, Ps, As = greedy_tree(
                            [S], [r], [tree], capacity.copy(), m, oddist, odpath
                        )
                        P = Ps[0]
                        A = As[0]
                    elif alg == "Camdoop":
                        tree = camdoop(G, S, r, nodes_each_edge)
                        cost, Ps, As = greedy_tree(
                            [S], [r], [tree], capacity.copy(), m, oddist, odpath
                        )
                        P = Ps[0]
                        A = As[0]
                    elif alg == "Avalanche":
                        tree = avra(G, S.copy(), r)
                        cost, Ps, As = greedy_tree(
                            [S], [r], [tree], capacity.copy(), m, oddist, odpath
                        )
                        P = Ps[0]
                        A = As[0]
                    else:
                        assert False
                    solution.append((cost, len(A)))
                else:
                    solution.append((last_costs[index], last_used[index]))
                if not silent:
                    print(alg, cost, P, A)

                if cost < last_costs[index]:
                    last_costs[index] = cost
                    last_used[index] = len(A)
                else:
                    enoughs[index] = True

            result_file.write(f"{list(itertools.chain.from_iterable(solution))}\n")
        result_file.write("\n")
        used = time.perf_counter() - start
        print(
            f"{topo} used:{used/60} mins, estimated:{used*(number_of_experiments-expno-1)/60} mins"
        )
        print()
    result_file.close()


if __name__ == "__main__":
    random.seed(322)
    compare_ideal("result.txt", "bcube", 2, False, False)
