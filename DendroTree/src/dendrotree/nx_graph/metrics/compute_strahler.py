import networkx as nx
import numpy as np
import sys


def num_nei(graph,node):
    return(len([v for v in graph[node]]))

def compute_strahler(graph,has_root = False):
    """
    returns a graph with ordered branch following strahler algorithm. If the graph has a root, then the orders flow from the root in a decreasing order
    also returns the lists of the branches (order,[root,...,end]) where root and end or nodes, order is the order of the branch
    """
    graph_copy = graph.copy()
    order = 1

    list_branches = []
    # for n in graph_copy:
    #     if num_nei(graph_copy,n) == 2:
    #         v,w = [z for z in graph_copy[n]]
    #         if graph_copy.has_edge(v,w):
    #             if np.concatenate([graph_copy[n][v]['coordinates'][1:-1],graph_copy[v][w]['coordinates'][1:-1],graph_copy[w][n]['coordinates'][1:-1]]).shape[0] == 0:
    #                 nodes_to_delete.append(n)

    # for n in nodes_to_delete:
    #         if n in graph_copy:
    #             graph_copy.remove_node(n)

    while not(nx.is_empty(graph_copy)):
        #print(graph_copy.number_of_nodes())
        #print([n for n in graph_copy])
        nodes_to_delete = []
        for n in graph_copy:
            if (num_nei(graph_copy,n) == 1) and (not(has_root) or not(graph_copy.nodes[n]["root"])):
                branch = [n]
                nodes_to_delete.append(n)
                v = [w for w in graph_copy[n]][0]
                u = n
                graph[n][v]['order'] = order
                branch.append(v)
                while num_nei(graph_copy,v) == 2:
                    nodes_to_delete.append(v)
                    z = list((set([w for w in graph_copy[v]]) - set([u])))[0]
                    graph[v][z]["order"] = order
                    branch.append(z)
                    u = v
                    v = z
                list_branches.append((order,branch[::-1]))

        if has_root:
            for n in graph_copy:
                if num_nei(graph_copy,n) == 1:
                    v = [w for w in graph_copy[n]][0]
                    if graph_copy.nodes[n]["root"] and graph_copy.nodes[v]["root"]:
                        nodes_to_delete.append(n)
                if num_nei(graph_copy,n) == 0:
                    nodes_to_delete.append(n)

                if num_nei(graph_copy,n) == 2:
                    v,w = [z for z in graph_copy[n]]
                    if graph_copy.nodes[v]["root"] and graph_copy.nodes[w]["root"]:
                        if graph_copy[n][v]["coordinates"][1:-1].shape[0] == 0 and graph_copy[n][w]["coordinates"][1:-1].shape[0] == 0:
                            graph_copy.nodes[v]["root"] = False


        if len(nodes_to_delete) == 0 and graph_copy.number_of_nodes() > 0:
            import pdb
            pdb.set_trace()

        for n in nodes_to_delete:
            if n in graph_copy:
                graph_copy.remove_node(n)
        order += 1
    list_branches = delete_doublons(list_branches)
    return(graph,list_branches)

def delete_doublons(list_branches):
    to_delete = []
    for i in range(len(list_branches)):
        for j in range(i+1,len(list_branches)):
            a = list_branches[i][1]
            b = list_branches[j][1]
            if np.all(np.array([x == y for (x,y) in zip(a,b)])) or np.all(np.array([x == y for (x,y) in zip(a,b[::-1])])):
                to_delete.append(j)

    for x in sorted(to_delete)[::-1]:
        del list_branches[x]
    return(list_branches)




            