"""
hypy1d - hyperbolic differential equations with python in 1d
Stefan Vater (2013)

grid representations
"""

import numpy as np


def generate_uniformgrid(xmin=0, xmax=1, nlength=11, periodic=True):
  nodecoordinates = np.linspace(xmin, xmax, nlength)

  elength           = nlength-1
  elementnodes      = np.zeros((elength,2), dtype=int)
  elementnodes[:,0] = np.arange(0, nlength-1)
  elementnodes[:,1] = np.arange(1, nlength)

  nodeelements  = -np.ones((nlength,2), dtype=int)
  for nd in range(nlength):
    ndelmts = np.where(np.any(elementnodes==nd, axis=1))
    nodeelements[nd,:np.size(ndelmts)] = ndelmts[0]

  # treatment of boundary conditions
  if (periodic):
    nodeelements[ 0,1] = nodeelements[0,0]
    nodeelements[ 0,0] = nodeelements[-1,0]
    nodeelements[-1]   = nodeelements[0]

  else:
    nodeelements[ 0,1] = nodeelements[0,0]
    nodeelements[ 0,0] = -1
    nodeelements[-1,1] = -1

  return nodecoordinates, elementnodes, nodeelements


class Grid1D:
  """
  representation of a one dimensional grid

  Class members:
    elength         : number of elements
    nlength         : number of nodes
    elementnodes    : nodes of each element
    elementnodest   : topological nodes of each element (for periodic b.c.)
    nodeelements    : elements adjacent to each node
    inodes          : indices of inner nodes
    bnodes          : indices of boundary nodes
    inodeelements   : elements adjacent to each inner node
    bnodeelements   : element adjacent to boundary node
    elementneighbors: neighbor elements of element i
    nodecoordinates : coordinates of each node
    elementcenter   : coordinates of element centers
    elementwidth    : width of each element
  """

  def __init__(self, nodecoordinates, elementnodes, nodeelements):

    self.nodecoordinates = nodecoordinates
    self.elementnodes    = elementnodes
    self.elementnodest   = np.copy(elementnodes)
    self.nodeelements    = nodeelements

    self.nlength       = len(nodecoordinates)
    self.elength       = self.nlength-1
    self.elementcenter = (nodecoordinates[elementnodes[:,0]]+nodecoordinates[elementnodes[:,1]])/2.0
    self.elementwidth  = nodecoordinates[elementnodes[:,1]]-nodecoordinates[elementnodes[:,0]]

    self.elementneighbors = np.zeros((self.elength, 2), dtype=int)
    self.elementneighbors[:,0] = self.nodeelements[self.elementnodes[:,0],0]
    self.elementneighbors[:,1] = self.nodeelements[self.elementnodes[:,1],1]
    
    self.inodes = np.where(np.all(nodeelements>=0,axis=1))[0]
    self.bnodes = np.where(np.any([nodeelements[:,0]<0, nodeelements[:,1]<0],axis=0))[0]
    self.inodeelements  = nodeelements[self.inodes]
    self.bnodeelements  = nodeelements[self.bnodes]
    if (nodeelements[0,0]==nodeelements[-1,0] and nodeelements[0,1]==nodeelements[-1,1]):
      self.inodes        = self.inodes[:-1]
      self.inodeelements = self.inodeelements[:-1]
      self.elementnodest[-1,1] = self.elementnodest[0,0]

#nodecoordinates, elementnodes, nodeelements = generate_uniformgrid(nlength=17, periodic=False)

#print(Grid1D(nodecoordinates, elementnodes, nodeelements).nodecoordinates)

#generate_uniformgrid(nlength=17)