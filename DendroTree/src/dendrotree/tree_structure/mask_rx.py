import rustworkx as rx
import networkx as nx
import numpy as np
import skan
from skimage import morphology,io

def num_nei(graph,node):
    return(len([v for v in graph[node]]))

def compute_path_length(path_coordinates):
    """
    euclidian distance of a path of ordered pixels coordinates
    path_coordinates is a np.array of shape (n,2), where n is the number of pixels in the path
    """
    plus = path_coordinates[1:]
    minus = path_coordinates[:-1]
    #print(path_coordinates)
    dist = np.linalg.norm(plus-minus,axis = -1)
    total_length = np.sum(dist)
    
    #print(plus)
    #print(minus)
    #print(dist)
    #print(total_length)
    return(total_length)

def mask2nx(skeleton_img:np.array,open_skeleton :bool):
    """
    takes a skeletonized object and turns it into a graph
    the skeleton_img should represent a tree (hence have no loop)
    The pixels with value 2 represent contact and will be set to zero
    The pixels with value 1 represent normal branch
    The pixels with value 3 represents root(s)
    """
    if open_skeleton:
        skeleton_img[~np.isin(skeleton_img,[1,3])] = 0
    assert np.any(skeleton_img==3),ValueError("tree should have at least one root (pixel with value 3)")
    if np.any(skeleton_img == 3):
        list_roots = []
        root = np.nonzero(skeleton_img == 3)
        for i in range(root[0].shape[0]):
            root_coords = (root[0][i],root[1][i])
            #rr,cc = skdraw.disk(root_coords,3,shape = skeleton_img.shape)
            #skeleton_img[rr,cc] = 0
            list_roots.append(root_coords)


    skel_object = skan.Skeleton(morphology.skeletonize(skeleton_img))#,unique_junctions = True)#junction_mode = 'centroid')#
    
    g = nx.Graph()
    
    
    for n in range(skel_object.n_paths):
        path = skel_object.path_coordinates(n)
        u,v = tuple(path[0]),tuple(path[-1])
        g.add_node(u)
        g.add_node(v)
        if not g.has_edge(u,v) and (u!=v or path.shape[0] > 5):
            g.add_edge(u,v,idx = n,length = skel_object.path_lengths()[n],coordinates = path)
        else:
            #this part is for the case when two nodes are connected by more than one path. We add an artificial node in the middle of the path
            if (path.shape[0] == 2 and np.all(path[0] == path[-1])) or (path.shape[0] > 2):
                head = path.shape[0]//2
                w = tuple(path[head])
                g.add_edge(u,w,length = compute_path_length(path[:head+1]),coordinates = path[:head+1])
                g.add_edge(w,v,length = compute_path_length(path[head:]),coordinates = path[head:])

    #adding roots
    
    for n in g:
        g.nodes[n]['root'] = False
    
    if len(list_roots) == 1:
        global_root = list_roots[0]
    else:
        global_root = np.mean(np.array(list_roots),axis = 0)
        global_root = (global_root[0],global_root[1])
        for root_coords in list_roots:
            g.add_edge(global_root,root_coords,length = 0,coordinates = np.stack([global_root,root_coords]))

    g.nodes[global_root]["root"]=True
    for root_coords in list_roots:
        # g.add_node(root_coords)
        # for n in g:
        #     if n != root_coords:
        #         if np.linalg.norm(np.array(n) - np.array(root_coords)) < 4.5 and not(g.nodes[n]["root"]):
        #             g.add_edge(n,root_coords,length = 3,coordinates = np.stack([n,root_coords],axis = 0))

        
        if root_coords in g:
            g.nodes[root_coords]['root'] = True

        

    nodes_to_delete = []
    for n in g:
        if num_nei(g,n) == 2:
            v,w = [z for z in g[n]]
            if g.has_edge(v,w):
                if np.concatenate([g[n][v]['coordinates'][1:-1],g[v][w]['coordinates'][1:-1],g[w][n]['coordinates'][1:-1]]).shape[0] == 0:
                    nodes_to_delete.append(n)
            
            # elif has_root:
            #     if g.nodes[v]['root'] and g.nodes[w]['root']:
            #         if np.concatenate([g[n][v]['coordinates'][1:-1],g[w][n]['coordinates'][1:-1]]).shape[0] == 0:
            #             g.remove_edge(n,w)

    for n in nodes_to_delete:
            if n in g:
                g.remove_node(n)
    return(g,global_root)

def nx2rx(g,global_root) -> rx.PyDiGraph:
    rx_tree = rx.PyDiGraph()
    coords2idx = dict([])
    if not(global_root is None):
        coords2idx[global_root] = rx_tree.add_node({"position":global_root[0] + 1j*global_root[1]})
        for source,target in nx.dfs_edges(g,source = global_root):
            coords2idx[target] = rx_tree.add_child(coords2idx[source],{"position" : target[0] + 1j*target[1]},None)
    else:
        for u,v in g.edges:
            if not(v in coords2idx.keys()) and not(u in coords2idx.keys()) :
                coords2idx[u] = rx_tree.add_node({"position":u[0] + 1j*u[1]})
            if (u in coords2idx.keys()) and not(v in coords2idx.keys()):
                coords2idx[v] = rx_tree.add_child(coords2idx[u],{"position" : v[0] + 1j*v[1]},None)
            if (v in coords2idx.keys()) and not(u in coords2idx.keys()):
                coords2idx[v] = rx_tree.add_child(coords2idx[v],{"position" : u[0] + 1j*u[1]},None)
            if (v in coords2idx.keys()) and (u in coords2idx.keys()):
                if not(rx_tree.has_edge(coords2idx[u],coords2idx[v])) and not(rx_tree.has_edge(coords2idx[v],coords2idx[u])):
                    rx_tree.add_edge(coords2idx[u],coords2idx[v],{})
    return(rx_tree)



def mask2rx(skeleton_img:np.array) -> rx.PyDiGraph:

    g,global_root = mask2nx(skeleton_img,open_skeleton=True)
    rx_tree =  nx2rx(g,global_root)
    return(rx_tree)



if __name__ == "__main__":
    import tree_plot
    import tree_operations as to
    import matplotlib.pyplot as plt
    source_tif = r"D:\0000NeuronsData\clean_movies_1_min\class_I\23.11.23 221NG overnight 30s Roper_movie-1\ij_sift\neuron_structure.tif"
    mask = io.imread(source_tif)
    rx_tree = mask2rx(mask[mask.shape[0]//2])
    rx_tree = to.compute_root_distance(rx_tree,False)
    rx_tree = to.compute_subtree_length(rx_tree,False)
    rx_tree = tree_plot.add_radial_position(rx_tree,"subtree_n_nodes")
    #rx_tree = tree_plot.add_radial_position(rx_tree,None)
    rx_tree = tree_plot.electrostatic_correction(rx_tree,"plot_radial_position",100,100,100,1e-1)
    fig,ax = tree_plot.quickplotradial_tree(rx_tree)
    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    #ax.margins(0.1)
    ax.axis("off")
    plt.show()