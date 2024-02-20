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
    pass


if __name__ == "__main__":
    random.seed(14632)
    compare_ideal("result.txt", "bcube", 2, False, False)
