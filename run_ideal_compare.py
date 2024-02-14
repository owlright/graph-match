import random
from multiprocessing import Process
from inc.exps.compare_ideal import compare_ideal

if __name__ == "__main__":
  topos = ["torus", "bcube", "fattree", "random"]
  number_of_experiments = 40
  seed = 417855
  random.seed(seed)
  for topo in topos:
    print(topo)
    result_file_name = "exp2-" + topo + "-result.txt"
    p = Process(target=compare_ideal, args=(result_file_name, topo, number_of_experiments, False))
    p.start()
  p.join()
