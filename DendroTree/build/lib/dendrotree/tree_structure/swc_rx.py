import rustworkx as rx
import numpy as np 
from .tree_operations import find_root


def swc2rx(swc_str):
    """
    input is a string contained in a .swc file
    http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    """
    resulting_graph = rx.PyDiGraph()
    dict_swc2rx_idx = {}
    dict_parents = {}
    lines = swc_str.split('\n')
    additional_columns_name = []
    for l in lines:
        if l != "":
            if l[0] == "#":
                if "Columns" in l:
                    additional_columns_name = [a for a in l.split(" ")[9:] if a !=""]
                else:
                    pass
            else:
                compartment = l.split(" ")
                if compartment[0] == "":
                    compartment = compartment[1:]
                comp_id = int(compartment[0])
                struct_id = int(compartment[1])
                x = float(compartment[2])
                y = float(compartment[3])
                z = float(compartment[4])
                radius = float(compartment[5])
                parent_comp_id = int(compartment[6])

                payload = {
                    "comp_id" : comp_id,
                    "struct_id" : struct_id,
                    "position" : np.array([x,y,z]),
                    "radius" : radius,
                    "parent_comp_id" : parent_comp_id,
                }
                for i,name in enumerate(additional_columns_name):
                    if len(compartment)> 7+i:   
                        payload[name] = float(compartment[7+i])

                dict_parents[comp_id] = parent_comp_id
                dict_swc2rx_idx[comp_id] = resulting_graph.add_node(payload)

    for comp_id in dict_swc2rx_idx.keys():
        parent_comp_id = dict_parents[comp_id]
        if parent_comp_id != -1:
            resulting_graph.add_edge(dict_swc2rx_idx[parent_comp_id],dict_swc2rx_idx[comp_id],None)

    return(resulting_graph)

def encode_node(comp_id,parent_comp_id,node_payload):
    """
    node_payload is a dict with the attributes of the node. It should at least have a "position" key, and can also have a "radius" key as well as a "struct_id" key
    """
    struct_id = node_payload["struct_id"] if "struct_id" in node_payload.keys() else 0
    position = node_payload["position"]
    radius = node_payload["radius"] if "radius" in node_payload.keys() else 0
    if isinstance(position,np.complex128):
        position = (np.real(position),np.imag(position),0)
    return(f" {comp_id} {struct_id} {position[0]} {position[1]} {position[2]} {radius} {parent_comp_id}")

def rx2swc(digraph:rx.PyDiGraph):
    """
    input is a rx.Pydigraph. Nodes should have least have a position attribute
    output is a string that can be directly written to a .swc file
    """
    res = ""
    dict_rx2swc_idx = {}
    
    root_idx = find_root(digraph)#[node_idx for node_idx in digraph.node_indices() if len(digraph.predecessor_indices(node_idx)) == 0][0]
    comp_id_counter = 1
    res += encode_node(comp_id_counter,-1,digraph[root_idx])
    dict_rx2swc_idx[root_idx] = comp_id_counter
    

    list_edges = rx.dfs_edges(digraph,source=root_idx)
    for (source,target) in list_edges:
        comp_id_counter += 1
        parent_comp_id = dict_rx2swc_idx[source]
        dict_rx2swc_idx[target] = comp_id_counter
        line = encode_node(comp_id_counter,parent_comp_id,digraph[target])
        res += "\n" + line

    return(res)


if __name__ == "__main__":
    with open("test.swc","r") as f:
        swc = f.read()

    digraph = swc2rx(swc)
    swc2 = rx2swc(digraph)
    with open("test_res.swc","w") as f:
        f.write(swc2)


    import matplotlib.pyplot as plt
    from walk_rx import *
    from tree_operations import find_root,compute_root_distance,compute_subtree_length,make_coarse_graph

    digraph = compute_root_distance(digraph,False)
    digraph = compute_subtree_length(digraph,False)
    dict_walk = rx2walk(digraph,"subtree_length",["position","subtree_length","root_dist_nodes"])
    print(dict_walk["walk_digits"])
    plt.plot(dict_walk["root_dist_nodes"])
    plt.show()
    plt.plot(dict_walk["subtree_length"])
    plt.show()

    coarse = make_coarse_graph(digraph)
    coarse = compute_root_distance(coarse,True)
    coarse = compute_subtree_length(coarse,True)
    dict_walk = rx2walk(coarse,"coarse_subtree_length",["position","coarse_subtree_length","coarse_root_dist_nodes"])
    print(dict_walk["walk_digits"])
    plt.plot(dict_walk["coarse_root_dist_nodes"])
    plt.show()
    plt.plot(dict_walk["coarse_subtree_length"])
    plt.show()

