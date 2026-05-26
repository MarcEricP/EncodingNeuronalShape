import networkx as nx
import numpy as np

def coord_to_int(tup):
    return(int(tup[0]),int(tup[1]))

def num_nei(graph,node):
    return(len([v for v in graph[node]]))

def compute_distance_to_root(graph):
    """
    returns a graph with ordered branch folowing strahler algorithm. If the graph has a root, then the orders flow from the root in a decreasing order
    also returns the lists of the branches (order,[root,...,end]) where root and end or nodes, order is the order of the branch
    """
    root = None
    for n in graph.nodes:
        if graph.nodes[n]["root"]:
            root = n
    if root is None:
        raise(ValueError("the graph must have a root"))

    p = nx.shortest_path_length(graph,root,weight="length")

    for n in graph.nodes:
        try:
            graph.nodes[n]["root_dist"] = p[n]
        except :
            graph.nodes[n]["root_dist"] = np.inf

    for u in graph.nodes:
        for v in graph[u]:
            coordinates = graph[u][v]["coordinates"].astype(int)
            
            s = 1 - np.arange(coordinates.shape[0])/(coordinates.shape[0]-1)
            if u == coord_to_int(coordinates[0]):
                root_dist = s*graph.nodes[u]["root_dist"] + (1-s)*graph.nodes[v]["root_dist"]
            else:
                root_dist = (1-s)*graph.nodes[u]["root_dist"] + s*graph.nodes[v]["root_dist"]

            graph[u][v]["root_dist"] = root_dist

    return(graph)