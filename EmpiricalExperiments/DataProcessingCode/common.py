import numpy as np
import networkx as nx

## helper method for a baseline
def create_triangle_list(G):
    elist = list(G.edges())
    
    triangles = []
    for e in elist:
        # consider the elist to be in form i, j
        i, j = e
        # neigbors of i are all nodes k that appears in the list
        first_node_neighbors = set(G.neighbors(i))
        # same for node j
        second_node_neighbors = set(G.neighbors(j))

        # find intersection between those neighbors => triangle
        common_neighbors = list(first_node_neighbors & second_node_neighbors)
        
        for t in common_neighbors:
            curr_triangle = np.sort([i,j,t])
            triangles.append(curr_triangle)
    possible_ts = np.unique(triangles, axis=0)
    return possible_ts
