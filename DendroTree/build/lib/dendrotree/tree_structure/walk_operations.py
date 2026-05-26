import numpy as np
def sort_tree_rec(tree):
    """
    tree: np array that starts and ends at zero, and reaches zero nowher else
    """
    #print(tree.shape)
    assert tree.min() == 0,"minimal should be zero"
    assert tree[0] == 0 and tree[-1] == 0,"should start and end with zero"
    assert tree.shape[0]%2 == 1,"tree should have odd length"
    if tree.shape[0] == 1:
        return((0,tree))
    elif tree.shape[0]>= 3 :
        idx = np.sort(np.nonzero(tree == 1)[0])

        if idx.shape[0] <= 2:
            stree = tree[1:-1] - 1
            h,stree = sort_tree(stree)
            return((h+1,np.concatenate([[0],stree + 1,[0]])))
        else:
            subtrees = [tree[idx[i] : idx[i+1]+1] - 1 for i in range(idx.shape[0] - 1)]
            res_subtrees = [sort_tree(s) for s in subtrees]
            Y = np.array([a for a,_ in res_subtrees])
            idx_sort = np.argsort(Y)
            res_subtrees = [res_subtrees[i] for i in idx_sort]
    
            res_subtrees = [a for a in reversed(res_subtrees)]
            height = [a for a,_ in res_subtrees]
            tree_height = max(height) + 1

            tree = np.concatenate([[0]]+[b[:-1]+1 for _,b in res_subtrees]+[[1,0]])
            #print(tree)
            return((tree_height,tree))
        
def sort_tree(tree):
    tree = np.concatenate([[0],tree + 1,[0]])
    _,tree = sort_tree_rec(tree)
    return(tree[1:-1] - 1)

def get_steps(tree):
    """
    return steps
    """
    idx_max = np.nonzero((tree[1:-1] > tree[2:])*(tree[1:-1] > tree[:-2]))[0]
    idx_min = np.nonzero((tree[1:-1] < tree[2:])*(tree[1:-1] < tree[:-2]))[0]
    idx_min = np.concatenate([[0],idx_min])
    steps_plus = tree[idx_max + 1] - tree[idx_min + 1]
    return(steps_plus)