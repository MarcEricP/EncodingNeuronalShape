import networkx as nx
import numpy as np

def binned_length_mode(g:nx.Graph,mode:str,bin_edges):
    """
    is_incr: whether the value increases as you get away from the soma
    """

    binned_length = np.zeros((bin_edges.shape[0]-1,))

    for u,v,d in g.edges(data=True):
        metric = d[mode]
        # cum_length = np.arange(d["coordinates"].shape[0] + 1)/(d["coordinates"].shape[0])*d["length"]
        # pxl_length = cum_length[1:] - cum_length[:-1]
        pxl_length = d["length"]/(d["coordinates"].shape[0]-1)
        idx = np.searchsorted(bin_edges,(metric[1:] + metric[:-1])/2, side='right') - 1
        idx = np.clip(idx, 0, len(bin_edges) - 2)

        binned_length[idx] += pxl_length
    return(binned_length)
