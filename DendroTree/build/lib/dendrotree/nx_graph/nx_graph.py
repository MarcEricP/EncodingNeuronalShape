import skan
import networkx as nx
import numpy as np
from skimage import measure
from skimage.morphology import skeletonize
import io
import math
from collections import defaultdict, deque
import networkx as nx
import numpy as np
import copy
import rustworkx as rx
import itertools

def get_coord(path,graph):
    
    coords = []
    for i in range(len(path) - 1):
        temp_coord = graph[path[i]][path[i+1]]["coordinates"]

        if np.any((temp_coord[0] - np.array(path[i])) != 0):
            temp_coord = temp_coord[::-1]
        coords.append(temp_coord)

    for i in range(len(coords) - 1):
        coords[i+1] = coords[i+1][1:]
    
    coords = np.concatenate(coords)
    return(coords)    

def num_nei(graph,node):
    return(len([v for v in graph[node]]))

def mask2nx(skeleton_img,has_contact = False,has_root = False):
    
    """
    takes a skeletonized object and turns it into a graph
    has_contact: bool, if True, the pixels of skeleton_img should have value 1 for regular branch pixel and 3 for a pixel that is the root of a tree.
    """
    skeleton_img = copy.deepcopy(skeleton_img)
    if not(np.any(skeleton_img > 0)):
        g = nx.Graph()
        return(g)
    if has_contact :
        skeleton_img[~np.isin(skeleton_img,[1,3])] = 0
    if has_root:
        if np.any(skeleton_img == 3):
            list_roots = []
            root = np.nonzero(skeleton_img == 3)
            for i in range(root[0].shape[0]):
                root_coords = coord_to_int((root[0][i],root[1][i]))
                #rr,cc = skdraw.disk(root_coords,3,shape = skeleton_img.shape)
                #skeleton_img[rr,cc] = 0
                list_roots.append(root_coords)
        else : 
            has_root = False

    skel_object = skan.Skeleton(skeletonize(skeleton_img))#,unique_junctions = True)#junction_mode = 'centroid')#
    
    g = nx.Graph()
    
    
    for n in range(skel_object.n_paths):
        path = skel_object.path_coordinates(n)
        u,v = coord_to_int(tuple(path[0])),coord_to_int(tuple(path[-1]))
        g.add_node(u)
        g.add_node(v)
        if not g.has_edge(u,v) and (u!=v or path.shape[0] > 5):
            g.add_edge(u,v,idx = n,length = skel_object.path_lengths()[n],coordinates = path)
        else:
            #this part is for the case when two nodes are connected by more than one path. We add an artificial node in the middle of the path
            if (path.shape[0] == 2 and np.all(path[0] == path[-1])) or (path.shape[0] > 2):
                head = path.shape[0]//2
                w = coord_to_int(tuple(path[head]))
                g.add_edge(u,w,length = compute_path_length(path[:head+1]),coordinates = path[:head+1])
                g.add_edge(w,v,length = compute_path_length(path[head:]),coordinates = path[head:])

    #adding roots
    if has_root:
        all_root = []
        for n in g:
            g.nodes[n]['root'] = False
        for root_coords in list_roots:
            # g.add_node(root_coords)
            # for n in g:
            #     if n != root_coords:
            #         if np.linalg.norm(np.array(n) - np.array(root_coords)) < 4.5 and not(g.nodes[n]["root"]):
            #             g.add_edge(n,root_coords,length = 3,coordinates = np.stack([n,root_coords],axis = 0))

            
            if root_coords in g:
                g.nodes[root_coords]['root'] = True
                all_root.append(root_coords)
        for u,v in itertools.product(all_root,all_root):
            if u!=v and (u in g[v]):
                g.remove_edge(u,v)
        if len(all_root)>0:
            centroid = np.mean(np.array(all_root),axis = 0)
            coords_root = coord_to_int((centroid[0],centroid[1]))
            for n in all_root:
                g.nodes[n]["root"] = False
                g.add_edge(coords_root,n,length = 0,coordinates = np.stack([coords_root,n]))
            g.nodes[coords_root]["root"] = True
            

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

    if has_contact and has_root:
        for c in nx.cycle_basis(g):
            # for a in c :
            #     print(a, g.nodes[a]["root"])
            for i in range(len(c)):
                idx1 = i
                idx2 = i+1
                if idx2 == len(c):
                    idx2 = 0
                if not(g.nodes[c[idx1]]["root"] or g.nodes[c[idx2]]["root"]) or len(c) == 1:
                    g.remove_edge(c[idx1],c[idx2])
                    break

    return(g)
    
def add_soma_nodes(g:nx.Graph,soma:np.array):
    """
    find graph vertices that are in the soma, add vertice that links them all and is the centroid of all soma nodes
    return the updated graph as well as the list of all nodes that belong to the soma
    """
    soma_nodes = []
    for n in nx.nodes(g):
        x,y = coord_to_int(n)
        if len(soma_nodes) > 0 :
            min_dist = np.min(np.linalg.norm(np.array([x,y]) - np.array(soma_nodes),axis = -1))
        else : 
            min_dist = np.inf
        if soma[x,y] and min_dist > 3.5:
            # print((x,y,min_dist))
            soma_nodes.append(n)

    #find centroid
    M = measure.moments(soma,order = 1)
    centroid = coord_to_int((M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]))

    #adding edges with length 0 between the nodes in the soma
    for n in soma_nodes:
        g.add_edge(n,centroid,idx = None,length = 0,coordinates = np.array([n,centroid]))
    soma_nodes.append(centroid)

    return(g,soma_nodes)

def get_tips_and_junctions(graph):
    tips = []
    junctions = []
    for n in graph:
        if num_nei(graph,n) == 1:
            tips.append(n)
        else :
            junctions.append(n)
    return(tips,junctions)


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

def coord_to_int(tup):
    return(int(tup[0]),int(tup[1]))

def draw(graph,shape,mode = 'brut'):
    """
    Drawing the graph according to a specific tag (mode)
    """
    canvas = np.zeros(shape,dtype = 'float')
    for n in nx.nodes(graph):
        nei = list(nx.neighbors(graph,n))
        for v in nei :
            coord = np.floor(graph[n][v]['coordinates']).astype(int)
            #print(graph[n][v].keys())
            if not mode == 'brut':
                try:
                    canvas[coord[:,0],coord[:,1]] = graph[n][v][mode]
                except:
                    0
                    #pdb.set_trace()
            else :
                canvas[coord[:,0],coord[:,1]] = 1
    return(canvas)


def tip2tip_score(graph):
    """
    edges of the graph should have a "length" arguments
    """
    tips = [n for n in graph.nodes if num_nei(graph,n) == 1]
    for e in graph.edges:
        graph.edges[e]['tip2tip'] = 0
    p = {a[0]:a[1] for a in nx.shortest_path(graph,weight = 'length')}
    for i in range(len(tips)):
        if tips[i] in p.keys():
            for j in range(i+1,len(tips)):
                if tips[j] in p[tips[i]].keys():
                    path = p[tips[i]][tips[j]]
                    for k in range(len(path)-1):
                        graph.edges[(path[k],path[k+1])]['tip2tip'] += 1
    return(graph)


def get_directed_graph_from_root(graph,root):
    """
    gets a graph flowing down from the root.
    The graph should have a root
    """

    digraph = nx.dfs_tree(graph,source = root)

    for n in digraph.nodes:
        for key in graph.nodes[n].keys():
            digraph.nodes[n][key] = graph.nodes[n][key]

    for e in digraph.edges:
        for key in graph.edges[e].keys():
            digraph.edges[e][key] = graph.edges[e][key]
    return(digraph)

def get_root(graph):
    for n in graph:
        if graph.nodes[n]["root"]:
            root = n
    return(root)

def add_pixel_value(g:nx.Graph,img,name):
    """
    add a propriety "name" to each graph edge: 
    it is an array with the same dimensions as the propriety coordinates, 
    where the values in the new propriety corresponds to the pixel intensity 
    of the image at the corresponding location in coordinates
    """
    for u,v,d in g.edges(data=True):
        if "coordinates" in d.keys():
            g[u][v][name] = img[d["coordinates"][:,0],d["coordinates"][:,1]]
    return(g)

def refine_graph(g:nx.Graph,max_l):
    """
    max_l is the maximal length for a branch, any branch longer than max_l will be divided into int(np.ceil(length/max_l)) segments
    """
    g = copy.deepcopy(g)
    all_edges = [(u,v) for u,v in g.edges()]
    for u,v in all_edges:
        length = g[u][v]["length"]
        n_seg = int(np.ceil(length/max_l))
        if n_seg > 1:
            coords = g[u][v]["coordinates"]
            for i in range(1,n_seg):
                new_node_idx = i*coords.shape[0]//n_seg
                previous_node_idx = (i-1)*coords.shape[0]//n_seg
                new_coords = coords[previous_node_idx:new_node_idx+1]
                new_length = compute_path_length(new_coords)
                
                g.add_edge(coord_to_int(coords[previous_node_idx]),coord_to_int(coords[new_node_idx]),coordinates = new_coords,length=new_length)
                g.nodes[coord_to_int(coords[new_node_idx])]['root'] = False
                if i == n_seg - 1:
                    new_coords = coords[new_node_idx:]
                    new_length = compute_path_length(new_coords)
                    g.add_edge(coord_to_int(coords[new_node_idx]),coord_to_int(coords[-1]),coordinates = new_coords,length=new_length)

            g.remove_edge(u,v)
    return(g)


def rx2nx(rx_graph:rx.PyDiGraph) -> nx.Graph:
    g = nx.Graph()
    for source,target in rx_graph.edge_list():
        coordinates_source = (rx_graph[source]["position"][0],rx_graph[source]["position"][1])
        coordinates_target = (rx_graph[target]["position"][0],rx_graph[target]["position"][1])
        length = np.sqrt((coordinates_source[0] - coordinates_target[0])**2 + (coordinates_source[1] - coordinates_target[1])**2)
        g.add_edge(coordinates_source,coordinates_target,length = length)
        

    return(g)