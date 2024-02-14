import networkx as nx


def minitopo():
  a = 'a'
  b = 'b'
  c = 'c'
  d = 'd'
  e = 'e'
  f = 'f'
  g = 'g'
  r = 'r'
  h = 'h'
  w = 'w'
  pos = 'pos'
  G = nx.Graph([
      (a, 4),
      (b, 4),
      (d, 7),
      (8, 2),
      (c, 1),
      (e, 2),
      (f, 8),
      (r, 3),
      (2, 4),
      (2, 6),
      (1, 4),
      (3, h),
      (5, 7),
      (1, 5),
      (5, 3),
      (3, 6),
      (4, 7),
      (6, 7),
      (g, 6),
      (1, 8),
      (8, 3),
  ])
  nodes = {
      c: {
          'pos': (0, 1),
          'type': 'host'
      },
      g: {
          'pos': (4, 2),
          'type': 'host'
      },
      e: {
          'pos': (2, 1.5),
          'type': 'host'
      },
      f: {
          'pos': (1, 0),
          'type': 'host'
      },
      d: {
          'pos': (3, 3),
          'type': 'host'
      },
      a: {
          'pos': (0, 2),
          'type': 'host'
      },
      h: {
          'pos': (3, 0),
          'type': 'host'
      },
      r: {
          'pos': (4, 1),
          'type': 'host'
      },
      # w:{'pos':(2, 3),'type':'host'},
      1: {
          'pos': (1, 1),
          'type': 'switch'
      },
      2: {
          'pos': (2, 1),
          'type': 'switch'
      },
      4: {
          'pos': (1, 2),
          'type': 'switch'
      },
      5: {
          'pos': (2, 2),
          'type': 'switch'
      },
      b: {
          'pos': (1, 3),
          'type': 'host'
      },
      6: {
          'pos': (3, 2),
          'type': 'switch'
      },
      3: {
          'pos': (3, 1),
          'type': 'switch'
      },
      7: {
          'pos': (2, 3),
          'type': 'switch'
      },
      8: {
          'pos': (2, 0),
          'type': 'switch'
      },
  }
  G.add_nodes_from([(node, attr) for node, attr in nodes.items()])
  return G