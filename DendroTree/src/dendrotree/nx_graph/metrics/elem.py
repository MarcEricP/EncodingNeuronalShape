import networkx as nx


def total_length(g:nx.Graph):
    """
    returns the total length (in pixels) of a nx_graph
    """
    return(sum([a["length"] for _,_,a in g.edges(data=True)]))

def n_branch_points(g:nx.Graph):
    n_bp = 0
    for u in g:
        neighbours = [a for a in g[u]]
        if len(neighbours)>2:
            n_bp += 1

    return(n_bp)