import os
import json
import numpy as np
import numpy as np

def find_files(list_dataroots,list_names,exclude = []):
    """
    list_dataroots = list of paths to explore
    list_names = list of tuples, if each element of the tuple is in the a folder ex: ("test.tif","pred/test2.tif")
    returns: list of tuples of paths 
    """
    res = []
    for r in list_dataroots:
        for root,dirs,_ in os.walk(r):
            dirs[:] = [d for d in dirs if d not in exclude]
            for l in list_names:
                
                list_suffix = [os.path.join(root,suffix) for suffix in l]
                
                append = True
                for suffix in list_suffix:
                    if not(os.path.exists(suffix)):
                        append = False
                if append:
                    res.append(tuple([root] + list_suffix))

    return(res)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types 
    
    from https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
