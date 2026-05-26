import rustworkx as rx
import numpy as np


def find_root(digraph:rx.PyDiGraph) -> int:
    """
    returns the index of the root of the directed graph which is supposed to be a tree with a single root
    """
    list_roots = [node_idx for node_idx in digraph.node_indices() if len(digraph.predecessor_indices(node_idx)) == 0]
    assert len(list_roots) == 1,ValueError("graph should have exactly one root")
    root_idx = list_roots[0]
    return(root_idx)

def make_coarse_graph(digraph:rx.PyDiGraph) -> rx.PyDiGraph:
    """
    - prunes all the nodes with exactly one predecessor and exactly one successor
    - the node of the coarse graph are a subset of those of the digraph. They share the payload which are copied in a shallowed way
    - the payload of the edges of the coarse_graph is the list of all the nodes of the input graph linking both nodes of the edge
    """
    coarse_graph = digraph.copy()

            
    for (source,target) in coarse_graph.edge_list():
        coarse_graph.update_edge(source,target,[])
    indices = coarse_graph.node_indices()
    for node_idx in indices:
        if len(coarse_graph.predecessor_indices(node_idx)) == 1 and len(coarse_graph.successor_indices(node_idx)) == 1:
            #nodes with exactly one predecessor and one successor, we get rid of it
            source = coarse_graph.predecessor_indices(node_idx)[0]
            target = coarse_graph.successor_indices(node_idx)[0]
            coarse_graph.add_edge(source,target,coarse_graph.get_edge_data(source,node_idx) + [node_idx] + coarse_graph.get_edge_data(node_idx,target))
            coarse_graph.remove_node(node_idx)

    return(coarse_graph)

def compute_root_distance(digraph : rx.PyDiGraph,is_coarse_graph : bool) -> rx.PyDiGraph:
    """
    returns the modified graph: each node has two additional attributes in its payload dictionary:
        - "root_dist_nodes":
        - "root_dist_length":  added only if the nodes have a position attribute
    If the inputed graph is a coarse_graph, the added attributes are instead: 
        - "coarse_root_dist_nodes"
        - "coarse_root_dist_length": added only if the nodes have a position attribute
    This is to avoid the conflict in the shared node payload between the coarse_graph and the original graph
    """
    position_ok = np.all(np.array(["position" in digraph[node_idx].keys() for node_idx in digraph.node_indices()]))

    dist_nodes_str = "root_dist_nodes"
    dist_length_str = "root_dist_length"
    if is_coarse_graph:
        dist_nodes_str = "coarse_" + dist_nodes_str
        dist_length_str = "coarse_" + dist_length_str
    #for node_idx in digraph.node_indices():
    root_idx = find_root(digraph)
    digraph[root_idx][dist_nodes_str] = 0
    if position_ok:
        digraph[root_idx][dist_length_str] = 0

    list_edges = rx.dfs_edges(digraph,source = root_idx)
    for (source,target) in list_edges:
        digraph[target][dist_nodes_str] = digraph[source][dist_nodes_str] + 1
        if position_ok:
            digraph[target][dist_length_str] = digraph[source][dist_length_str] + np.linalg.norm(digraph[target]["position"] - digraph[source]["position"])
    return(digraph)

def compute_subtree_length(digraph:rx.PyDiGraph,is_coarse_graph : bool) -> rx.PyDiGraph:
    """
    returns the modified graph: each node has two additional attributes in its payload dictionary:
        - "subtree_n_nodes":
        - "subtree_length":  added only if the nodes have a position attribute
        - "subtree_lenth_upstream": added only if the nodes have a position attribute
    If the inputed graph is a coarse_graph, the added attributes are instead: 
        - "coarse_subtree_n_nodes"
        - "coarse_subtree_length": added only if the nodes have a position attribute
        - "coarse_subtree_length_upstream": added only if the nodes have a position attribute
    This is to avoid the conflict in the shared node payload between the coarse_graph and the original graph
    """
    position_ok = np.all(np.array(["position" in digraph[node_idx].keys() for node_idx in digraph.node_indices()]))

    dist_nodes_str = "subtree_n_nodes"
    dist_length_str = "subtree_length"
    dist_length_str_upstream = "subtree_length_upstream"
    if is_coarse_graph:
        dist_nodes_str = "coarse_" + dist_nodes_str
        dist_length_str = "coarse_" + dist_length_str
        dist_length_str_upstream = "coarse_" + dist_length_str_upstream

    for node_idx in digraph.node_indices():

        digraph[node_idx][dist_nodes_str] = np.nan
        if position_ok: 
            digraph[node_idx][dist_length_str_upstream] = np.nan
    # take the terminal tips
    queue = [node_idx for node_idx in digraph.node_indices() if len(digraph.successor_indices(node_idx)) == 0]

    while len(queue) > 0:
        #print(len(queue))
        current_node_idx = queue[0]
        queue = queue[1:]

        current_successors = digraph.successor_indices(current_node_idx)
        if len(current_successors) == 0:
            #terminal tips

            digraph[current_node_idx][dist_nodes_str] = 0
            
            if position_ok:
                digraph[current_node_idx][dist_length_str] = 0

            #upstream length
            if len(digraph.predecessor_indices(current_node_idx)) > 0:
                if position_ok:
                    digraph[current_node_idx][dist_length_str_upstream] = np.linalg.norm(digraph[digraph.predecessor_indices(current_node_idx)[0]]["position"] - digraph[current_node_idx]["position"])
            else:
                if position_ok :
                    digraph[current_node_idx][dist_length_str_upstream] = 0

            
        else:
            digraph[current_node_idx][dist_nodes_str] = len(current_successors) + np.sum(np.array([digraph[temp_succ][dist_nodes_str] for temp_succ in current_successors]))
            if position_ok:
                list_subtree_length_successors = np.array([digraph[temp_succ][dist_length_str_upstream] for temp_succ in current_successors])
                # if np.any(np.isnan(list_subtree_length_successors)):
                #     print("c")
                #     #queue.append(current_node_idx)
                # else:
                #     print("d")
                # list_branch_id_successor = [digraph[temp_succ]["subtree_length_branch_id"] for temp_succ in current_successors]
                # max_sub_tree_idx = np.argmax(list_subtree_length_successors)
                # chosen_branch_id = list_branch_id_successor[max_sub_tree_idx]
                
                digraph[current_node_idx][dist_length_str] = np.sum(list_subtree_length_successors)
                digraph[current_node_idx][dist_length_str_upstream] = digraph[current_node_idx][dist_length_str]
                if len(digraph.predecessor_indices(current_node_idx)) > 0:
                    digraph[current_node_idx][dist_length_str_upstream] += np.linalg.norm(digraph[digraph.predecessor_indices(current_node_idx)[0]]["position"] - digraph[current_node_idx]["position"])
                
        
        # add the parent of the node if it is ready to be considered
        if len(digraph.predecessor_indices(current_node_idx)) > 0:
            parent_node = digraph.predecessor_indices(current_node_idx)[0]
            current_successors = digraph.successor_indices(parent_node)
            list_subtree_nodes_successors = np.array([digraph[temp_succ][dist_nodes_str] for temp_succ in current_successors])
            if not(np.any(np.isnan(list_subtree_nodes_successors))):
                queue.append(parent_node)
    return(digraph)
