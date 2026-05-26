import numpy as np
import networkx as nx
from skimage import morphology,measure
import itertools
from ..nx_graph import nx_graph

#function with soma arguments can be used as such by passing a image equals to zero everywhere for the soma 


def graph_from_cycle(cycles):
    graph2 = nx.Graph() 
    for (j,cycle) in zip(range(len(cycles)),cycles):
        for i in range(len(cycle) - 1):
            graph2.add_edge(cycle[i],cycle[i+1])
            graph2[cycle[i]][cycle[i+1]]['cycle'] = j + 1
        graph2.add_edge(cycle[-1],cycle[0])
        graph2[cycle[-1]][cycle[0]]['cycle'] = j + 1
    return(graph2)

def copy_edges_keys(graph_target,graph_source):
    for e in graph_target.edges:
        for key in graph_source.edges[e].keys():
            graph_target.edges[e][key] = graph_source.edges[e][key]
    return(graph_target)

def order_cycles(cycles,graph_source):
    #### orders the cycles
    cycles_ordered = []
    for cyc in cycles : 
        temp_graph = graph_source.subgraph(cyc)
        temp_cyc = nx.cycle_basis(temp_graph)
        cycles_ordered.append(temp_cyc[0])
    return(cycles_ordered)
    
def delete_cycles_in_soma(cycles,soma_nodes):
    #delete cycles that are completely in the soma
    temp_cycles = []
    for c in cycles:
        keep = False
        for n in c:
            if not(n in soma_nodes):
                keep = True
        if keep:
            temp_cycles.append(c)
    return(temp_cycles)

def delete_minimal_cycles(cycles,graph_source,soma_nodes,avoid_intersection = True):
    to_del = []
    for j in range(len(cycles)):
        c = cycles[j]
        coord = []
        for i in range(len(c) - 1):
            if not(c[i] in soma_nodes and c[i+1] in soma_nodes):
                if avoid_intersection:
                    coord.append(graph_source[c[i]][c[i+1]]['coordinates'][1:-1])
                else:
                    coord.append(graph_source[c[i]][c[i+1]]['coordinates'])

        if not(c[-1] in soma_nodes and c[0] in soma_nodes):
            if avoid_intersection:
                coord.append(graph_source[c[-1]][c[0]]['coordinates'][1:-1])
            else:
                coord.append(graph_source[c[-1]][c[0]]['coordinates'])

        coord = np.concatenate(coord)

        if coord.shape[0] == 0:
            to_del.append(j)
    to_del = sorted(to_del)[::-1]
    for a in to_del:
        del cycles[a]
    return(cycles)


def find_pixels_to_cut(cycles_ordered,graph_source,age_mask,soma_nodes,avoid_intersection = True):
    """
    
    """
    to_cut = []
    true_minimum = [] #states that the corresponfing pixel to cut is indeed a minimum
    cut_value = []
    
    coord = []
    for k in range(len(cycles_ordered)) :
        c = cycles_ordered[k]
        
        coord.append([])
        for i in range(len(c) - 1):
            if not(c[i] in soma_nodes and c[i+1] in soma_nodes):
                if avoid_intersection:
                    coord[-1].append(graph_source[c[i]][c[i+1]]['coordinates'][1:-1])
                else:
                    coord[-1].append(graph_source[c[i]][c[i+1]]['coordinates'])

        if not(c[-1] in soma_nodes and c[0] in soma_nodes):
            if avoid_intersection:
                coord[-1].append(graph_source[c[-1]][c[0]]['coordinates'][1:-1])
            else:
                coord[-1].append(graph_source[c[-1]][c[0]]['coordinates'])

        coord[-1] = np.concatenate(coord[-1])

        coord_int = coord[-1].astype(int)

        coord_age = age_mask[coord_int[:,0],coord_int[:,1]]
        try :
            min_idx = np.argmin(coord_age)
        except :
            import pdb
            pdb.set_trace()
            
        if not(np.any(coord_age > np.min(coord_age)) or coord_age.shape[0] == 1):
            true_minimum.append(False)
        else:
            true_minimum.append(True)

        to_cut.append(coord_int[min_idx])
        cut_value.append(coord_age[min_idx])
    
    to_cut = np.array(to_cut)
    cut_value = np.array(cut_value)
    true_minimum = np.array(true_minimum)
    
    
    return(to_cut,cut_value,true_minimum)

def delete_point_from_graph(coord_point,graph:nx.Graph):
    to_del = dict([])
    to_add = []
    for e in graph.edges:
        coordinates = graph.edges[e]['coordinates'].astype(int)
        dist = np.linalg.norm(coordinates - coord_point,axis = 1)
        idx_cut = np.argmin(dist)
        if dist[idx_cut] == 0:
            to_del[e] = True
            u = nx_graph.coord_to_int(coordinates[0])
            v = nx_graph.coord_to_int(coordinates[-1])
            u1 = nx_graph.coord_to_int(coordinates[idx_cut - 1])
            v1 = nx_graph.coord_to_int(coordinates[idx_cut + 1])
            if u != u1:
                to_add.append((u,u1,coordinates[:idx_cut]))
            if v != v1:
                to_add.append((v,v1,coordinates[idx_cut+1:]))
                
            
        else:
            to_del[e] = False
            
    for e in to_del.keys():
        if to_del[e]:
            graph.remove_edge(*e)
    
    for (u,v,coordinates) in to_add:
        graph.add_edge(u,v,length = nx_graph.compute_path_length(coordinates),coordinates=coordinates)
    
    return(graph)


def get_cut_points(age_mask_idx,graph,soma_nodes,skel = None,soma = None,true_minimum_only = True,avoid_intersection = True):
    """
    if true_minimum_only is true, we are not satisfied whith a random point in an egal cycle
    """
    #age dilation
    age_dil =   age_mask_idx#morphology.dilation(age_mask_idx,footprint = morphology.disk(5))

    #### finds minimal cycles
    cycles = nx.cycle_basis(graph)
    graph2 = graph_from_cycle(cycles)
    graph2 = copy_edges_keys(graph2,graph)
    cycles = nx.minimum_cycle_basis(graph2,weight = 'length')

    #### orders the cycles
    cycles_ordered = order_cycles(cycles,graph2)
    #delete cycles that are completely in the soma
    cycles_ordered = delete_cycles_in_soma(cycles_ordered,soma_nodes)
    cycles_ordered = delete_minimal_cycles(cycles_ordered,graph2,soma_nodes,avoid_intersection = avoid_intersection)

    max_iter = len(cycles_ordered)
    num_iter = 0

    res_cut_point = []

    while len(cycles_ordered) > 0 and num_iter < 1000:
        # print(num_iter)
        # if num_iter > 100:
        #     import pdb
        #     pdb.set_trace()
        #print(num_iter)
        #canvas = canvas = gh.draw(graph2,skel.shape,mode = 'brut')
        #viewer.add_image(uf.dil(canvas))



        num_iter += 1
        ### find the pixels in the skeleton where to cut
        to_cut_all,cut_value_all,true_minimum_all = find_pixels_to_cut(cycles_ordered,graph2,age_dil,soma_nodes,avoid_intersection=avoid_intersection)
        #print(cut_value)

        if not(true_minimum_only):
            true_minimum_all = np.ones(true_minimum_all.shape[0],dtype = bool)

        to_cut = to_cut_all[true_minimum_all]
        cut_value = cut_value_all[true_minimum_all]

        if np.any(true_minimum_all):
            temp_idx = np.argmin(cut_value)
            #print(cut_value[idx])
            cut_point = to_cut[temp_idx]

            res_cut_point.append(cut_point)

            # delete edges from graph
            if skel is None:
                graph2 = delete_point_from_graph(cut_point,graph2)
            else :
                skel[int(cut_point[0]),int(cut_point[1])] = 0
                graph2 = nx_graph.mask2nx(skel)
                graph2,_ = nx_graph.add_soma_nodes(graph2,soma)
                # graph2,_ = make_graph(morphology.skeletonize(skel > 0),soma)

            cycles = nx.minimum_cycle_basis(graph2,weight = 'length')


            #### orders the cycles
            cycles_ordered = order_cycles(cycles,graph2)

            #delete cycles that are completely in the soma
            cycles_ordered = delete_cycles_in_soma(cycles_ordered,soma_nodes)

            cycles_ordered = delete_minimal_cycles(cycles_ordered,graph2,soma_nodes,avoid_intersection = avoid_intersection)
        else :
            cycles_ordered = []
    return(np.array(res_cut_point))
    

def expand_cut_points(skel,age_mask_t,cut_points):
    res_cut_points = []

    for k in range(cut_points.shape[0]):
        cut = cut_points[k]
        res_cut_points.append(tuple(cut))
        age_cut = age_mask_t[cut[0],cut[1]]

        file = [tuple(cut)]
        seen = [tuple(cut)]

        while len(file) > 0:
            cut_2 = file[0]
            file = file[1:]
            for i,j in itertools.product([-1,0,1],[-1,0,1]):
                coords = (cut_2[0] + i,cut_2[1] + j)
                if not(i == 0 and j == 0) and (skel[coords[0],coords[1]] > 0) and (age_mask_t[coords[0],coords[1]] - age_cut < 0.9) and not(coords in seen):
                    res_cut_points.append(coords)
                    file.append(coords)
                    seen.append(coords)
    return(np.array(res_cut_points).astype(int))


