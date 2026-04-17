from transforms3d.euler import euler2quat
from transforms3d import quaternions
import numpy as np

def _get_absolute_matrix(node, nodes_dict):
    current_matrix = np.array(node[2]["matrix"])
    parent_name = node[0]
    while parent_name != "world":
        parent_node = nodes_dict[parent_name]
        parent_matrix = np.array(parent_node[2]["matrix"])
        current_matrix = parent_matrix @ current_matrix
        parent_name = parent_node[0]
    return current_matrix

def _get_pq(matrix, origin):
    matrix = np.array(matrix)
    q = quaternions.mat2quat(matrix[:3,:3])
    p = matrix[:-1, 3] - origin
    return p, q
