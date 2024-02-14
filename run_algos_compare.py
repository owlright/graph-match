
import random
from multiprocessing import Process
from inc.exps.compare_algo import compare_algo

if __name__ == "__main__":
  topos = ["torus", "bcube", "fattree", "random"]
  number_of_experiments = 20
  seed = 417855
  random.seed(seed)
  for topo in topos:
    print(topo)
    result_file_name = "exp1-" + topo + "-result.txt"
    p = Process(target=compare_algo, args=(result_file_name, topo, number_of_experiments, False))
    p.start()
  p.join()
