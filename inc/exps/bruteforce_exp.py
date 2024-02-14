import itertools
import numpy as np
import math
import multiprocessing
from multiprocessing import Pool, shared_memory
import signal
import traceback
import time
import array
import os

try: # ! when you run from main.py
  from ..algorithms.extract_paths_cython import extract_paths
  from ..algorithms.extract_paths import extract_paths as extract_paths_nb
  from ..util import do_cprofile
except ImportError: # ! when you run here
  import sys, os
  sys.path.append(os.getcwd())
  from inc.util import *
  from inc.algorithms.extract_paths import extract_paths as extract_paths_nb
  from inc.algorithms.extract_paths import extract_paths

DTYPE = np.int32
global minCost
global minA
global minP


def heavy_comp(A):
  # print(A)
  thispid = str(os.getppid())
  tmp = shared_memory.SharedMemory(name=thispid+'otherinfo')
  otherinfo = np.ndarray((4, ), dtype=DTYPE, buffer=tmp.buf)
  Slen = otherinfo[0]
  a = otherinfo[1]
  b = otherinfo[2]
  r = otherinfo[3]
  tmp1 = shared_memory.SharedMemory(name=thispid+'S')
  S = np.ndarray((Slen,), dtype=DTYPE, buffer=tmp1.buf )
  tmp2 = shared_memory.SharedMemory(name=thispid+'oddist')
  oddist = np.ndarray((a, b), dtype=DTYPE, buffer=tmp2.buf)

  # r = context['r']
  # oddist = context['dist']
  try:
    mstnodes = np.array([r]+list(A), dtype=DTYPE)
    cost = extract_paths(S, mstnodes, oddist)
    # sindexes = list(P.keys())
    # tindexes = [P[s] for s in sindexes]
    # cost = np.sum(oddist[sindexes, tindexes])
    # return A, dict(P)
    return A, cost
  except:
    traceback.print_exc()

# @do_cprofile("./bruteforce.prof")
# ! this function can only deal with a single tree
def bruteforce_exp(S, r, capacity, M, oddist, Debug=False):
  forbiddens = set()
  last_cost = 0
  # ! remove nodes cannot be computation nodes
  for node in capacity:
    if capacity[node] == 0:
      forbiddens.add(node)
  # ! get potential nodes (which is W in my paper) for each tree
  paggrs = set(capacity) - forbiddens # * burte force use all possible nodes
  cost = np.sum(oddist[S, r])
  last_cost += cost
  Sarr = np.array(S, dtype=DTYPE)

  minCost = last_cost
  minA = None
  minP = None
  thispid = str(os.getpid())

  tmp1 = shared_memory.SharedMemory(size=Sarr.nbytes, create=True, name=thispid+'S')
  S_share = np.ndarray(Sarr.shape, Sarr.dtype, buffer=tmp1.buf)
  S_share[:] = Sarr
  tmp3 = shared_memory.SharedMemory(size=oddist.nbytes, create=True, name=thispid+'oddist')
  oddist_share = np.ndarray(oddist.shape, oddist.dtype, buffer=tmp3.buf)
  oddist_share[:, :] = oddist[:, :]

  pieces = np.array([len(Sarr), oddist.shape[0], oddist.shape[1], r], dtype=DTYPE)

  tmp4 = shared_memory.SharedMemory(create=True, size=pieces.nbytes, name=thispid+'otherinfo')
  otherinfo = np.ndarray(pieces.shape, dtype=pieces.dtype, buffer=tmp4.buf)
  otherinfo[:] = pieces[:]

  ncombs = math.comb(len(paggrs), M)
  print(f"M: {M}, W: {len(paggrs)} number of combinations: {ncombs}")

  pool = multiprocessing.Pool()
  suggest_chunk, extra = divmod(ncombs, len(pool._pool) * 4)
  if suggest_chunk == 0:
    suggest_chunk = extra
  chunksize = suggest_chunk if suggest_chunk < 1e6 else int(1e6)
  results = pool.imap(heavy_comp, itertools.combinations(paggrs, M), chunksize=chunksize)
  for A, cost in results:
    if cost < minCost:
      minCost = cost
      minA = A
  pool.close()
  pool.join()

  tmp1.close()
  tmp1.unlink()
  tmp3.close()
  tmp3.unlink()
  tmp4.close()
  tmp4.unlink()

  minP = extract_paths_nb(Sarr, np.array([r]+list(minA), dtype=DTYPE), oddist)
  return minCost, minP, minA
