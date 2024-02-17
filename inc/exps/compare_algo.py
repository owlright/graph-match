import time
from .exp_setup import *
from ..topo import *

# from ..algorithms.tree import *
from ..algorithms import *
from ..util import plot


def compare_algo(
    result_file_name: str,
    topo: nx.Graph,
    number_of_experiments,
    Debug=True,
    silent=True,
):
    result_file = open(result_file_name, "w")
    parameters = ["res", "cap", "nTree", "nS"]
    tree_algos = ["shortest", "steiner"]
    greedy_algos = ["Shortest", "Takashami", "GCAT"]
    if topo == "fattree":
        greedy_algos.append("Avalanche")
        G_origin = fattree(16)
    elif topo == "random":
        N = random.randint(300, 400)
        G_origin = random_graph(N)
    elif topo == "torus":
        greedy_algos.append("Camdoop")
        G_origin = torus(8)
    elif topo == "bcube":
        greedy_algos.append("IRS")
        G_origin = bcube(3, 4)
    else:
        raise ValueError("wrong topo name" + topo)
    G_reindexed, _ = get_reindexed_graph(G_origin)
    odpath, oddist = get_paths(G_reindexed)
    hosts = get_attr_nodes(G_reindexed, "type", "host")
    table_names = parameters + tree_algos + greedy_algos

    table_border, table_header_format, table_format = get_table_format(
        len(parameters), len(tree_algos), len(greedy_algos)
    )
    # result_file.write(str(tree_algos))
    # result_file.write('\n')
    result_file.write(str(greedy_algos))
    result_file.write("\n")
    result_file.write(topo)
    result_file.write("\n")

    richness = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    capacity_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # richness = [0.5, 1]
    # capacity_ratio = [0.5, 1]

    result_file.write(f"{richness}\n{capacity_ratio}\n{number_of_experiments}\n")

    # ! step 1. generate topos
    if not silent:
        print("=" * 26 + topo + "=" * 26)
    # result_file.write(topo + "\n")

    exp_time = []
    for expno in range(number_of_experiments):
        start = time.process_time()
        if not silent:
            print(f"ExpSeq: {expno}")
            print(table_border)
            print(table_header_format.format(*table_names))
            print(table_border)

        for treeshare_ratio in richness:
            for swichcapacito_ratio in capacity_ratio:
                # todo: is random necessary? since S and r are already random

                # number_of_trees = 10
                # number_of_sources = 50

                number_of_trees = random.randint(20, 50)
                number_of_sources = random.randint(50, 100)

                # * max resource a job can use
                maxM = number_of_trees * number_of_sources * swichcapacito_ratio

                # ! Only random topo can change each time
                # todo is this necessary?
                if topo == "random":
                    N = random.randint(300, 400)
                    G_origin = random_graph(N)
                    G_reindexed, _ = get_reindexed_graph(G_origin)
                    hosts = get_attr_nodes(G_reindexed, "type", "host")
                    # get od matrix
                    odpath, oddist = get_paths(G_reindexed)
                # * switch capacity is determined by switch capacity ratio
                set_capacity(G_reindexed, int(number_of_trees * swichcapacito_ratio))
                # * Prepare each tree's sources and targets

                # ! I want to print the two special algos' results
                sources = [
                    random.sample(hosts, number_of_sources + 1)
                    for _ in range(number_of_trees)
                ]
                targets = [S.pop() for S in sources]
                if topo != "bcube":
                    steiner_trees = [
                        build_steiner_tree(S, r, oddist, odpath, set(hosts))
                        for S, r in zip(sources, targets)
                    ]
                else:
                    steiner_trees = [
                        build_steiner_tree(S, r, oddist, odpath)
                        for S, r in zip(sources, targets)
                    ]
                shortestpath_trees = [
                    build_sptree(S, r, odpath) for S, r in zip(sources, targets)
                ]

                base_solution = [
                    list(get_treesolution(sources, targets, shortestpath_trees)),
                    list(get_treesolution(sources, targets, steiner_trees)),
                ]

                memory_canused = int(treeshare_ratio * maxM)
                capacity = nx.get_node_attributes(G_reindexed, "capacity")
                # * first let we try different original trees
                # prepare_no_inc_solution(non_inc_solutions, number_of_trees, steiner_trees, shortestpath_trees, targets)
                # ! traffic, used resource
                solutions = [[0, 0] for _ in range(len(greedy_algos))]
                # ! A0, merged_trees, capacity must not be changed during an experiment
                for algo in greedy_algos:
                    if algo == "GCAT":
                        traffic, Ps, As = gstep(
                            sources,
                            targets,
                            capacity.copy(),
                            memory_canused,
                            oddist,
                            odpath,
                        )
                    else:
                        if algo == "Shortest":  # * tree algos
                            trees = shortestpath_trees
                        elif algo == "Takashami":
                            trees = steiner_trees
                        elif algo == "Camdoop":
                            trees = [
                                camdoop(G_reindexed, S, r, 8)
                                for S, r in zip(sources, targets)
                            ]
                        elif algo == "IRS":
                            trees = [
                                irs_based(G_reindexed, S, r, 3, False)
                                for S, r in zip(sources, targets)
                            ]
                        elif algo == "Avalanche":
                            trees = [
                                avra(G_reindexed, S.copy(), r)
                                for S, r in zip(sources, targets)
                            ]
                        else:
                            assert False
                        traffic, Ps, As = greedy_tree(
                            sources,
                            targets,
                            trees,
                            capacity.copy(),
                            memory_canused,
                            oddist,
                            odpath,
                        )
                    # ! actual resources algorithm used because
                    # ! when M is redunant some resource will not be used.
                    num_of_comp_nodes = sum(
                        [len(As[k]) for k in range(number_of_trees)]
                    )

                    if Debug:
                        for k in range(number_of_trees):
                            st = steiner_trees[k]
                            st_aggrs = [n for n in st if st.in_degree[n] >= 2]
                            # print(k, len(As[k]), len(st_aggrs))
                            T = construct_tree_from_paths(Ps[k], odpath)
                            nx.set_node_attributes(T, "switch", "type")
                            nx.set_node_attributes(T, "white", "color")
                            T.add_nodes_from(sources[k] + [targets[k]], type="host")
                            T.add_nodes_from(As[k], color="cyan")
                            num_edges = T.number_of_edges()
                            num_aggrs = sum([1 for n in T if T.in_degree(n) >= 2])
                            print()

                    solutions[greedy_algos.index(algo)][0] = traffic
                    solutions[greedy_algos.index(algo)][1] = num_of_comp_nodes
                table_content = tuple(
                    [
                        treeshare_ratio,
                        swichcapacito_ratio,
                        number_of_trees,
                        number_of_sources,
                    ]
                    + list(itertools.chain.from_iterable(base_solution + solutions))
                )
                if not silent:
                    print_results = table_format.format(*table_content)
                    print(print_results)
                result_file.write(f"{table_content}\n")
        if not silent:
            print(table_border)
            print()
        used_time = time.process_time() - start
        print(f"{topo} used time(min): {used_time / 60:.3f}")
        exp_time.append(used_time)
        print(
            f"{topo} rest time(min): {(sum(exp_time) / len(exp_time)) *(number_of_experiments - 1 - expno) / 60:.3f}"
        )
    result_file.close()
