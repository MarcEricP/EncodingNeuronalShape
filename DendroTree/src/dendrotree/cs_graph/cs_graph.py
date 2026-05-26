import tqdm
import skan
import numpy as np
from scipy.sparse import csgraph
import tqdm

def to_cs_graph(skeleton_img):
    graph,coordinates = skan.csr.skeleton_to_csgraph(skeleton_img)
    coordinates = np.stack(coordinates,axis = 1)
    distance,predecessor = csgraph.shortest_path(graph,return_predecessors=True)
    intersections = np.diff(graph.indptr)
    intersections = np.nonzero(intersections > 2)[0]
    return(graph,distance,predecessor,coordinates,intersections)

def movie_to_cs_graph(skeleton_movie):
    list_graph = []
    list_distance = []
    list_predecessor = []
    list_coordinates = []
    list_intersections = []
    for t in tqdm.trange(skeleton_movie.shape[0]):
        graph,distance,predecessor,coordinates,intersections = to_cs_graph(skeleton_movie[t])
        list_predecessor.append(predecessor)
        list_coordinates.append(coordinates)
        list_graph.append(graph)
        list_distance.append(distance)
        list_intersections.append(intersections)

    dico = {
            'list_graph' : list_graph,
            'list_coordinates' : list_coordinates,
            'list_distance' : list_distance,
            'list_predecessor' : list_predecessor,
            'list_intersections' : list_intersections
            }
    
    return(dico)

def project_on_cs_graph(points,coordinates,return_dist = False):
    """
    points of shape (n,2)
    coordinates of shape (m,2)
    """
    dist = np.linalg.norm(np.expand_dims(points,axis = 1) - np.expand_dims(coordinates,axis = 0),axis = -1)
    idx = np.argmin(dist,axis = 1)
    proj_coordinates = coordinates[idx]
    dist = dist[np.arange(dist.shape[0]),idx]
    if return_dist:
        return(idx,proj_coordinates,dist)
    else:
        return(idx,proj_coordinates)

def get_cs_graph_path(point_a,point_b,predecessor):
    path = [point_a]
    if predecessor[point_a,point_b] == -9999:
        path = []
    else:
        while path[-1]!=point_b :
                        
            path.append(predecessor[point_b,path[-1]])
    path = path[::-1]
    return(path)
