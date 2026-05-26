import rustworkx as rx
from typing import Tuple,List
from typing import List,Tuple,Dict
from itertools import compress
import numpy as np
from .tree_operations import find_root

def rx2walk(digraph:rx.PyDiGraph,mode : str,attributes : List[str]) -> Dict[str,List]:
    """
    attributes: list of attributes to return a walk for. The attributes have to be keys of every node's payload. "position" is a special attribute because of its dimension
    if mode is:
        - "subtree_length"/"coarse_subtree_length": the successors node with the maximal subtree length is (depth-first) visited first etc.
        - Depr: the set of trees is not totally ordered "subtree_n_nodes"/"coarse_subtree_n_nodes": the successors node with the maximal subtree number of nodes is (depth-first) visited first etc.
        - "RLRS_sequence", then the walks are totally ordered
    be careful when computing the excursion_walk whether or not you inputed a coarse_graph and whether or not this is what you wanted
    
    Returns a dict of list, each list corresponding to an attribute, the element "walk_nodes" and "walk_digits" of the dictionary are the representations of the excursion walk
    if a digit is True, then we went down a not yet visited child of the preceding_node, if it is False, we went back to the parent of the preceding node.
    """

    root_idx = find_root(digraph)
    excursion_list_nodes_idx = []
    excursion_list_digits = []
    pile = [root_idx]
    for node_idx in digraph.node_indices():
        digraph[node_idx]["excursion_visit_left"] = len(digraph.successor_indices(node_idx)) + 1
    
    preceding_node_idx = -1
    while len(pile) > 0:
        #print(len(pile))
        
        current_node_idx = pile.pop()
        #if digraph[current_node_idx]["excursion_visit_left"] > 0:
        
        excursion_list_nodes_idx.append(current_node_idx)
        
        if preceding_node_idx == -1:
            excursion_list_digits.append(True)
        else:
            if digraph.has_edge(preceding_node_idx,current_node_idx):
                excursion_list_digits.append(True)
            elif digraph.has_edge(current_node_idx,preceding_node_idx):
                excursion_list_digits.append(False)


        digraph[current_node_idx]["excursion_visit_left"] -= 1        
        
        if digraph[current_node_idx]["excursion_visit_left"] == 0:
            predecessors = digraph.predecessor_indices(current_node_idx)

            pile += list(predecessors)
        else:
            successors = digraph.successor_indices(current_node_idx) 
            successors_sub_tree_upstream = [digraph[succ][mode] for succ in successors]
            # elif mode == "n_nodes_priority":
            # if mode == "subtree_length_priority":
            #     successors_sub_tree_upstream = [digraph[succ]["subtree_length_upstream"] for succ in successors]
            # elif mode == "n_nodes_priority":
            #     successors_sub_tree_upstream = [digraph[succ]["subtree_n_nodes"] for succ in successors]

            successors_visitable = [digraph[succ]["excursion_visit_left"] > 0 for succ in successors]

            successors = list(compress(successors, successors_visitable))
            successors_sub_tree_upstream = list(compress(successors_sub_tree_upstream, successors_visitable))

            successors = [succ for (_,succ) in sorted(zip(successors_sub_tree_upstream,successors))]
            if len(successors) > 0:
                pile.append(successors[-1])

        preceding_node_idx = current_node_idx

        #print(digraph[current_node_idx]["position"],digraph[current_node_idx]["excursion_visit_left"],excursion_list_digits)

    dict_walk = {
        "walk_nodes" : excursion_list_nodes_idx,
        "walk_digits" : excursion_list_digits,
    }
    for attr in attributes:
        dict_walk[attr] = [digraph[node_idx][attr] for node_idx in excursion_list_nodes_idx]
            
    return(dict_walk)


def walk2rx(dict_walk:Dict[str,List],enforce_valid_walk:bool)-> rx.PyDiGraph:
    """
    returns a tuple (valid_walk,rustworx PyDiGraph built from the elements in dict_walk). If valid_walk is False, it means that the walk did not stop at the root, hence that a part of the graph is missing
    One key of dict_walk has to be "walk_digits". The first digit is ignored by convention
    The other keys will be handled as attributes and added to the payload of the resulting graph
    enforce_valid_walk: if True, the walk is truncated i such a way that it starts and end in the same node, without creating new node
    """


    digraph = rx.PyDiGraph()
    digits = dict_walk["walk_digits"]
    max_range = len(digits)
    if enforce_valid_walk:
        cumulative_sum = np.cumsum(2*np.array(digits,dtype = int) - 1) - 1
        nonz = np.nonzero(cumulative_sum < 0)[0]
        if nonz.shape[0] > 0:
            max_range = nonz[0]  
    valid_walk = True

    root_payload = {key:dict_walk[key][0] for key in  dict_walk.keys() if not(key == "walk_digits")}
    root_idx = digraph.add_node(root_payload)
    preceding_node_idx = root_idx
    for i in range(1,max_range):

        if digits[i]:
            #we go down a not yet visited child of the preceding_node
            child_payload = {key:dict_walk[key][i] for key in  dict_walk.keys() if not(key == "walk_digits")}
            preceding_node_idx = digraph.add_child(preceding_node_idx,child_payload,None)
        else:
            # we go back to the parent of the preceding node.
            predecessor = digraph.predecessor_indices(preceding_node_idx)
            
            if len(predecessor) == 1:
                #we visited the predecessor already, the tree is still valid, we just go to the predecessor
                preceding_node_idx = predecessor[0]
            elif len(predecessor) == 0:
                #the tree is invalid ! we never visited the predecessor of the node that we try to visit. We create it and signal that the tree is invalid
                valid_walk = False
                parent_payload = {key:dict_walk[key][i] for key in  dict_walk.keys() if not(key == "walk_digits")}
                preceding_node_idx = digraph.add_parent(preceding_node_idx,parent_payload,None)
            else:
                raise(ValueError("the constructed graph is not a tree, which should not occur by construction of this program"))

    return(valid_walk,digraph)


if __name__ == "__main__":
    from tree_operations import compute_subtree_length,make_coarse_graph,compute_subtree_RLRS,find_root
    import matplotlib.pyplot as plt
    from RLRS_trees import rlrs2rx
    digraph = rx.PyDiGraph()
    root_idx = digraph.add_node({"position" : np.array([0,0,0])})
    list_idx = [root_idx]
    pred = root_idx
    for i in range(5):
        pred = digraph.add_child(pred,{"position" : np.array([i+1,0,0])/100},None)
        list_idx.append(pred)
    pred = root_idx
    for i in range(2):
        pred = digraph.add_child(pred,{"position" : np.array([0,i+1,0])},None)
        list_idx.append(pred)


    digraph = compute_subtree_length(digraph,False)
    dict_walk = rx2walk(digraph,"subtree_n_nodes",["position"])
    assert np.all(np.array(dict_walk["walk_digits"]) == np.array([True,True,True,True,True,True,False,False,False,False,False,True,True,False,False]))

    coarse = make_coarse_graph(digraph)
    coarse = compute_subtree_length(coarse,True)

    dict_walk = rx2walk(coarse,"coarse_subtree_n_nodes",["position"])
    assert np.all(np.array(dict_walk["walk_digits"]) == np.array([True,True,False,True,False]))


    pred = list_idx[-5]
    for i in range(5):
        pred = digraph.add_child(pred,{"position" : digraph[list_idx[3]]["position"] + np.array([i+1,0,0])/100},None)
        list_idx.append(pred)
    digraph = compute_subtree_length(digraph,False)
    dict_walk = rx2walk(digraph,"subtree_length",["position","subtree_length"])
    print(dict_walk["walk_digits"])
    # plt.plot(np.cumsum(2*np.array(dict_walk["walk_digits"],dtype = int) - 1) - 1)
    # plt.show()
    # plt.plot(dict_walk["subtree_length"])
    # plt.show()


    pred = list_idx[-6]
    for i in range(4):
        pred = digraph.add_child(pred,{"position" : digraph[list_idx[3]]["position"] + np.array([i+1,0,0])/100},None)
        list_idx.append(pred)
    pred = digraph.add_child(list_idx[-3],{"position" : digraph[list_idx[3]]["position"] + np.array([i+1,0,0])/100},None)
    list_idx.append(pred)

    pred = list_idx[-2]
    for i in range(4):
        pred = digraph.add_child(pred,{"position" : digraph[list_idx[3]]["position"] + np.array([i+1,0,0])/100},None)
        list_idx.append(pred)
    pred = digraph.add_child(list_idx[-2],{"position" : digraph[list_idx[3]]["position"] + np.array([i+1,0,0])/100},None)
    list_idx.append(pred)

    pred = list_idx[-2]
    for i in range(2):
        pred = digraph.add_child(pred,{"position" : digraph[list_idx[3]]["position"] + np.array([i+1,0,0])/100},None)
        list_idx.append(pred)
    pred = digraph.add_child(list_idx[-2],{"position" : digraph[list_idx[3]]["position"] + np.array([i+1,0,0])/100},None)
    list_idx.append(pred)
    pred = digraph.add_child(pred,{"position" : digraph[list_idx[3]]["position"] + np.array([i+1,0,0])/100},None)
    list_idx.append(pred)
    pred = digraph.add_child(pred,{"position" : digraph[list_idx[3]]["position"] + np.array([i+1,0,0])/100},None)
    list_idx.append(pred)

    digraph = make_coarse_graph(digraph)
    digraph = compute_subtree_RLRS(digraph)
    root = find_root(digraph)
    rlrs = digraph[root]["RLRS_sequence"]
    print(rlrs.rlrs)
    digraph2 = rlrs2rx(rlrs)
    digraph2 = compute_subtree_RLRS(digraph2)

    dict_walk = rx2walk(digraph,"RLRS_sequence",[])
    #print(dict_walk["walk_digits"])
    plt.plot(np.cumsum(2*np.array(dict_walk["walk_digits"],dtype = int) - 1) - 1)

    dict_walk2 = rx2walk(digraph2,"RLRS_sequence",[])
    #print(dict_walk["walk_digits"])
    plt.plot(np.cumsum(2*np.array(dict_walk2["walk_digits"],dtype = int) - 1) - 1)
    plt.show()



