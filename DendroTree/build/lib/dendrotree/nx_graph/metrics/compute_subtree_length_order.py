import networkx as nx
import numpy as np
def coord_to_int(tup):
    return(int(tup[0]),int(tup[1]))

def num_nei(graph,node):
    return(len([v for v in graph[node]]))

def compute_subtree_length(graph,has_root = False):
    """
    returns a graph with ordered branch folowing strahler algorithm. If the graph has a root, then the orders flow from the root in a decreasing order
    also returns the lists of the branches (order,[root,...,end]) where root and end or nodes, order is the order of the branch
    """
    graph_copy = graph.copy()

    while not(nx.is_empty(graph_copy)):
        #print(graph_copy.number_of_nodes())
        #print([n for n in graph_copy])
        nodes_to_delete = []
        for n in graph_copy:
            if (num_nei(graph_copy,n) == 1) and (not(has_root) or not(graph_copy.nodes[n]["root"])):
                nodes_to_delete.append(n)
                v = [w for w in graph_copy[n]][0]

                down_stream_length = 0
                for w in graph[n]:
                    if not(w in graph_copy):
                        down_stream_length += graph[n][w]['sub_tree_length']
                graph[n][v]['sub_tree_length'] = graph[n][v]["length"] + down_stream_length

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
        
    for u in graph.nodes:
        for v in graph[u]:

            max_l_u = max([graph[u][n]["sub_tree_length"] for n in graph[u]])
            max_l_v = max([graph[v][n]["sub_tree_length"] for n in graph[v]])
            if max_l_u > max_l_v:
                graph.nodes[u]["subtree_length_detail"] = graph[u][v]["sub_tree_length"]
                graph.nodes[v]["subtree_length_detail"] = graph[u][v]["sub_tree_length"] - graph[u][v]["length"]
            else:
                graph.nodes[v]["subtree_length_detail"] = graph[u][v]["sub_tree_length"]
                graph.nodes[u]["subtree_length_detail"] = graph[u][v]["sub_tree_length"] - graph[u][v]["length"]
            coordinates = graph[u][v]["coordinates"].astype(int)
            
            s = 1 - np.arange(coordinates.shape[0])/(coordinates.shape[0]-1)
            if u == coord_to_int(coordinates[0]):
                subtree_length_detail = s*graph.nodes[u]["subtree_length_detail"] + (1-s)*graph.nodes[v]["subtree_length_detail"]
            else:
                subtree_length_detail = (1-s)*graph.nodes[u]["subtree_length_detail"] + s*graph.nodes[v]["subtree_length_detail"]
            graph[u][v]["subtree_length_detail"] = subtree_length_detail


    return(graph)

def compute_subtree_length_branches_centrifuge(graph,has_root = True):
    
    S = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    
    list_branch = []
    for G in S:
        seeds = []
        if has_root:
            true_root = True
            for n in G:
                if G.nodes[n]['root']:
                    seeds.append(n)
        if len(seeds) == 0:
            true_root = False
            all_nodes = []
            for n in G:
                max_l = 0
                max_node = None
                for m in G[n]:
                    if G[n][m]["sub_tree_length"] > max_l:
                        max_l = G[n][m]["sub_tree_length"]
                        max_node = m
                if not(max_node is None):
                    all_nodes.append((n,max_l))

            lengths = np.array([l for _,l in all_nodes])
            idx = np.argmax(lengths)
            seeds.append(all_nodes[idx][0])
        
        order = 1

        to_process = [(n,1) for n in seeds]
        processed_nodes = []

        while len(to_process) > 0:
            n,current_order = to_process[0]
            to_process = to_process[1:]
            processed_nodes.append(n)
            current_branch = [n]
            
            do = True
            redo_branch = True
            already_done_once = False
            while(redo_branch):
                redo_branch = False
                if already_done_once:
                    current_branch = current_branch[::-1]
                    do = True
                while do:
                    
                    current_node = current_branch[-1]
                    max_l = -1
                    max_node = None
                    
                    for m in G[current_node]:
                        if not(m in current_branch or m in processed_nodes):
                            try:
                                if G[current_node][m]["sub_tree_length"] > max_l:
                                    max_l = G[current_node][m]["sub_tree_length"]
                                    max_node = m
                            except:
                                0
                    if not(max_node is None):
                        current_branch.append(max_node)
                        processed_nodes.append(max_node)
                        graph[current_node][max_node]["order"] = current_order
                    else:
                        do = False

                    if not(true_root) and (current_node in seeds) and not(already_done_once):
                        redo_branch = True
                        already_done_once = True

            
            if len(current_branch) > 1:
                list_branch.append((current_order,current_branch))

            for m in current_branch:
                for w in G[m]:
                    if not(w in processed_nodes or w in current_branch):
                        to_process.append((m,current_order + 1))

    return(graph,list_branch)
