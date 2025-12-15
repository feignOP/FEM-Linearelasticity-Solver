"""
-----------------------------------------------
Written by: Anand Mathew
-----------------------------------------------
"""

# Opening Rituals
import numpy as np
from typing import Tuple

def create_id_matrix(constraints: np.ndarray, dofs_per_nodes: int, NumNodes: int) -> Tuple[np.ndarray, int]:
  """
   Function that provides equation and constraint numbers for a given set of (finite element) nodes.

  Parameters
  ----------
  constraints : np.ndarray
      Array of shape (Ncons, 3) with [node #, dof #, value]
  dofs_per_nodes : int
      Number of degrees of freedom per node.
  NumNodes : int
      Total number of nodes.

  Returns
  -------
  Global_ID : np.ndarray
      gives equation or constraint number
      Positive numbers = equation numbers ,
      Negative numbers = constraint numbers.
  NEqns : int
      Total number of free equations.

  Raises
  ------
  ValueError
      If inputs are invalid or inconsistent.
  """

  # Basic input validation
  if not isinstance(constraints, np.ndarray):
        raise ValueError("Error: 'constraints' must be a NumPy array.")
  if not isinstance(dofs_per_nodes, int) or dofs_per_nodes <= 0:
      raise ValueError("Error: 'dofs_per_nodes' must be a positive integer.")
  if not isinstance(NumNodes, int) or NumNodes <= 0:
      raise ValueError("Error: 'NumNodes' must be a positive integer.")
  if constraints.ndim != 2 or constraints.shape[1] != 3:
      raise ValueError(f"Error: 'constraints' must have shape (Ncons, 3). Got shape {constraints.shape} instead.")
  
  # Checking duplicate (node, dof) pairs
  node_mum = constraints[:, 0].astype(int)
  dof_mum  = constraints[:, 1].astype(int)

  pairs = np.column_stack((node_mum, dof_mum))
  unique_pairs = np.unique(pairs, axis=0)

  if len(unique_pairs) != len(pairs):
      raise ValueError("Error: Duplicate (node, dof) pairs found in the constraints.")
  
  # Initializing
  Global_ID = np.zeros((NumNodes, dofs_per_nodes), dtype=int)
  NEqns = 0
  Ncons = constraints.shape[0] 

  for i in range(Ncons):
    tnode = constraints[i, 0]
    tdof = constraints[i, 1]
    node = int(tnode)
    dof = int(tdof)

    # Cecking the type
    if abs(tnode - node) > 1e-15:
        raise ValueError(f"Error: node must be integer-valued at constraint {i+1} (got {tnode}).")
    if abs(tdof - dof) > 1e-15:
        raise ValueError(f"Error: dof must be integer-valued at constraint {i+1} (got {tdof}).")
    

    # Node validity
    if node < 1 or node > NumNodes:
        raise ValueError(f"Error: Invalid node number, {node}, at constraint {i+1}.")

    # DOF validity
    if dof < 1 or dof > dofs_per_nodes:
        raise ValueError(f"Error: Invalid DOF number, {dof}, at constraint {i+1}.")
    
    Global_ID[node-1, dof-1] = -(i+1)

  # Assigning IDs
  for i in range(NumNodes):
    for j in range(dofs_per_nodes):
      if Global_ID[i, j] <0:
        continue
      NEqns = NEqns + 1
      Global_ID[i, j] = NEqns

  return Global_ID, NEqns
