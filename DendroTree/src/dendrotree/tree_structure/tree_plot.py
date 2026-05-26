import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib
import rustworkx as rx
import numpy as np
from . import tree_operations as to
import tqdm


def add_vertical_position(rx_graph:rx.PyDiGraph,priority_mode):
    """
    For plotting purpose: adds a "plot_radial_position" attribute to each node such that the root of the tree is at the bottom 
    and each node has position (x, node height). Each subtree gets a part of the avalaible space proportionate to its width 
    (maximal number of nodes in the subtree that have the same height)
    priotiy_mode can be None or any attribute such as "subtree_length". Siblings are ordered by highest value.
    """

def add_node_label(rx_graph:rx.PyDiGraph,priority_mode,list_node_label,no_leaf_label = True):
    """
    priotiy_mode can be None or any attribute such as "subtree_length". Siblings are ordered by highest value.
    list_node_label should be a list of string, the nodes in the tree will get an attribute "node_label" with 
    value the element in in list_node_label in the order of visit of the tree fowllonwing the priority_mode
    """
    root = to.find_root(rx_graph)
    pile = [root]
    iter_list_node_label = 0
    while len(pile) > 0:
        current_node = pile.pop()
        
        list_successors = np.array(list(rx_graph.successor_indices(current_node)))

        if not(list_node_label is None):
            if not(no_leaf_label) :
                rx_graph[current_node]["node_label"] = list_node_label[iter_list_node_label]
                iter_list_node_label += 1
            elif list_successors.shape[0]>0:
                rx_graph[current_node]["node_label"] = list_node_label[iter_list_node_label]
                iter_list_node_label += 1
            else:
                rx_graph[current_node]["node_label"] = ""

        if list_successors.shape[0] > 0:
            if not(priority_mode is None):
                priority_value = np.array([rx_graph[node_idx][priority_mode] for node_idx in list_successors])
                idx_sort = np.argsort(priority_value)#[::-1]
                list_successors = list_successors[idx_sort]

            pile += [a for a in list_successors]
    return(rx_graph)

def add_radial_position(rx_graph:rx.PyDiGraph,priority_mode):
    """
    For plotting purpose: adds a "plot_radial_position" attribute to each node such that the root of the tree is at the center 
    Each subtree of an node gets an equal space
    priotiy_mode can be None or any attribute such as "subtree_length". Siblings are ordered by highest value.
    if list_node_label is not None, it should be a list of string, the nodes in the tree will get an attribute "node_label" with 
    value the element in in list_node_label in the order of visit of the tree fowllonwing the priority_mode
    """
    root = to.find_root(rx_graph)
    rx_graph[root]["plot_radial_position"] = 0
    rx_graph[root]["plot_radial_angle_limits_and_height"] = (0,2*np.pi,0)
    pile = [root]
    iter_list_node_label = 0
    while len(pile) > 0:
        current_node = pile.pop()
        
        list_successors = np.array(list(rx_graph.successor_indices(current_node)))

        if list_successors.shape[0] > 0:
            if not(priority_mode is None):
                priority_value = np.array([rx_graph[node_idx][priority_mode] for node_idx in list_successors])
                idx_sort = np.argsort(priority_value)[::-1]
                list_successors = list_successors[idx_sort]

            min_angle,max_angle,height = rx_graph[current_node]["plot_radial_angle_limits_and_height"]
            angle_split = (max_angle - min_angle)/len(list_successors)
            for i,node_idx in enumerate(list_successors):
                temp_mina,temp_maxa,temp_h = (min_angle + i*angle_split,min_angle + (i+1)*angle_split,height + 1)
                rx_graph[node_idx]["plot_radial_angle_limits_and_height"] = (temp_mina,temp_maxa,temp_h)
                rx_graph[node_idx]["plot_radial_position"] = temp_h*np.exp(1j*(temp_mina + temp_maxa)/2)

            pile += [a for a in list_successors]
    return(rx_graph)
        


def plot_position(rx_graph:rx.PyDiGraph,ax,position_mode,color_mode=None,cmap=plt.cm.viridis,color_min=None,color_max=None,node_label = False,plot_root=True):
    """
    position_mode can be:
    - "position"
    - "plot_radial_position" (see add_vertical_position)
    - "plot_vertical_position" (see add_radial_position)
    color_mode can be None or any attribute shared by all the nodes in the graph (for example "root_dist_nodes")
    node_label can be True of False, if True, every node in the graph should have attribute "node_label" with a string that will be printed next to the node
    cmap is None a matplotlib colormap (for example plt.cm.viridis)
    color_min,color_max are None or the value of the minimal and maximal values for color_mode
    scatter_bp: add a point at position of each node of the graph
    """
    assert np.all(np.array([position_mode in rx_graph[node_idx].keys() for node_idx in rx_graph.node_indices()])),ValueError(f"Some nodes do not have a {position_mode} attribute")
    line_collection = []
    line_value = []
    node_positions = []
    dict_node_label = dict([])
    # for node_idx in rx_graph.node_indices():
    #     node_positions.append((np.real(rx_graph[node_idx][position_mode]),np.imag(rx_graph[node_idx][position_mode])))
    # node_positions = np.array(node_positions)
    # print("hello")
    for source,target in rx_graph.edge_list():
        source_x,source_y = np.real(rx_graph[source][position_mode]),np.imag(rx_graph[source][position_mode])
        target_x,target_y = np.real(rx_graph[target][position_mode]),np.imag(rx_graph[target][position_mode])
        line_collection.append([( source_x,source_y),(target_x,target_y)])
        if node_label:
            dict_node_label[source] = ((source_x,source_y),rx_graph[source]["node_label"])
            dict_node_label[target] = ((target_x,target_y),rx_graph[target]["node_label"])
        if not(color_mode is None):
            line_value.append(rx_graph[target][color_mode])
    if cmap is None:
        cmap = plt.cm.viridis
    # print("aa")
    if len(line_collection)>0:
        if not(color_mode is None):
            line_value = np.array(line_value)
            local_min,local_max = np.min(line_value),np.max(line_value)
            if not(color_min is None):
                local_min = color_min
            if not(color_max is None):
                local_max = color_max
            line_value = (line_value - local_min)/(local_max - local_min + 1e-10)*1.0
            line_color = cmap(line_value)
            #print(line_value)
            lc = matplotlib.collections.LineCollection(line_collection, colors=line_color, linewidths=2,capstyle = "round")
        else :
            if isinstance(cmap,str):
                lc = matplotlib.collections.LineCollection(line_collection, colors = cmap, linewidths=2,capstyle = "round")
            else:
                lc = matplotlib.collections.LineCollection(line_collection, colors = [cmap(1) for _ in line_collection], linewidths=2,capstyle = "round")
        # print("aaa")
        ax.add_collection(lc)
    # print("bb")
    # if scatter_bp:
    #     ax.scatter(node_positions[:,0],node_positions[:1])
    if node_label and len(dict_node_label) > 0:
        for (x, y), lbl in dict_node_label.values():
            ax.annotate(
                str(lbl),
                xy=(x, y),
                xytext=(4, 3),    
                textcoords="offset points",
                ha="left", va="bottom",
                zorder=3, clip_on=True
            )

    cmap = matplotlib.cm.get_cmap('viridis')
    if plot_root:
        rgba = cmap(0.)
        root = to.find_root(rx_graph)
        root_coord = rx_graph[root][position_mode]
        ax.scatter([np.real(root_coord)],[np.imag(root_coord)],c=rgba,zorder=2)
    return(ax)

def quickplotradial_tree(rx_tree:rx.PyDiGraph):
    fig,ax = plt.subplots(1,1)
    ax = plot_position(rx_tree,ax,"plot_radial_position","root_dist_nodes",plt.cm.viridis,None,None)
    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    ax.margins(0.1)
    return(fig,ax)

if __name__ == "__main__":

    import special_trees
    n_nodes = 510
    #rx_tree = special_trees.cayley_tree(n_nodes,2)
    #rx_tree = special_trees.height_1_tree(n_nodes)
    #rx_tree = special_trees.random_walk_tree(n_nodes)
    #rx_tree = special_trees.reflected_random_walk_tree(n_nodes)
    #rx_tree = special_trees.random_binary_tree(n_nodes)
    rx_tree = special_trees.random_binary_tree(n_nodes,1,0)
    rx_tree = to.make_coarse_graph(rx_tree)
    rx_tree = to.compute_root_distance(rx_tree,False)
    rx_tree = to.compute_subtree_length(rx_tree,False)

    rx_tree = add_radial_position(rx_tree,None)
    #rx_tree = add_radial_position(rx_tree,"subtree_n_nodes")

    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1 = plot_position(rx_tree,ax1,"plot_radial_position","root_dist_nodes",plt.cm.viridis,None,None)
    ax1.autoscale()
    ax1.set_aspect('equal', adjustable='box')
    ax1.margins(0.1)
    #plt.colorbar()
    #plt.show()
    # rx_tree = electrostatic_correction(rx_tree,"plot_radial_position",100,100,100,1e-1)
    #fig,ax = plt.subplots()
    ax2 = plot_position(rx_tree,ax2,"plot_radial_position","root_dist_nodes",plt.cm.viridis,None,None)
    ax2.autoscale()
    ax2.set_aspect('equal', adjustable='box')
    ax2.margins(0.1)
    #plt.colorbar()
    plt.show()
            