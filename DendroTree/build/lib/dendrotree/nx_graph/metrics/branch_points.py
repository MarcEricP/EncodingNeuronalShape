import networkx as nx


def n_branch_points(g:nx.Graph):
    n_bp = 0
    for u in g:
        neighbours = [a for a in g[u]]
        if len(neighbours)>2:
            n_bp += 1

    return(n_bp)