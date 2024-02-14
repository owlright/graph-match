import math
import numpy as np
from util.utils import *


def D(n, m):
  ways = np.zeros((n+1, m+1))
  for i in crange(1, n):
    for j in crange(1, m):
      if i==0 or i > j:
        ways[i,j]=0
      elif i==1:
        ways[i,j]=1
      else:
        for k in crange(1, j-i+1):
          ways[i, j] += math.comb(j, k)*ways[i-1,j-k]
  return ways[n, m]

def L(c, k): 
  '''We denote Lc as the number of ways a ToR switch goes up to n ASs, 
  where only c of them are allowed to leave the pod.'''
  for i in crange(c, k/2): # since c up-edges, at least 
    num_of_aggr_choices = math.comb(k/2, i)
    math.comb(i, c)
  
print(D(2,3))