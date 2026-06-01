import numpy as np 
import pandas
from . import useful as uf
import tqdm
import os
import pickle

def load_dataset_from_scattered_dirs(dataset_path,filter = "flow_track_opened_skel"):
    dataset = dict([])

    list_paths = [a for a in uf.find_files([dataset_path],[("length_over_time.csv",)]) if filter in a[0]]

    for root,path in tqdm.tqdm(list_paths):
        split_path = root.split(os.sep)
        neuron_class = split_path[-5]
        neuron_name = split_path[-4]
        branch_num = split_path[-1]
        if not(neuron_class in dataset.keys()):
            dataset[neuron_class] = dict([])
        if not(neuron_name in dataset[neuron_class].keys()):
            dataset[neuron_class][neuron_name] = dict([])
        dataset[neuron_class][neuron_name][int(branch_num)] = dict([])
        track = pandas.read_csv(path)

        for c in track.columns:
            dataset[neuron_class][neuron_name][int(branch_num)][c] = np.array(track[c])

    return(dataset)

def filter_tracks(dataset,max_incr,min_len,contact_dist):
    """
    max_incr: if an increment is greater than max_incr in absolute value, the track is split at this point
    min_len: resulting tracks are kept only if they are longer than min_len
    contact dist: if not None, the tracks will be split around contact and contact_dist is the time to wait after a contact to admit the branch back
    """
    for n_class in dataset.keys():
        for n_name in dataset[n_class].keys():
            list_initial_keys = list(dataset[n_class][n_name].keys())
            for branch in list_initial_keys:
                time = dataset[n_class][n_name][branch]["time"]
                length = dataset[n_class][n_name][branch]["length"]
                is_contact = dataset[n_class][n_name][branch]["is_contact"]
                increments = length[1:] - length[:-1]
                xx = np.nonzero(np.abs(increments) > max_incr)[0]
                if xx.shape[0] > 0:
                    xx = np.concatenate([[0],xx+1,[length.shape[0] - 1]])
                    max_branch = max(b for b in dataset[n_class][n_name].keys())
                    for i in range(xx.shape[0]-1):
                        if xx[i+1] - xx[i] > min_len:
                            max_branch += 1
                            dataset[n_class][n_name][max_branch] = {key:value[xx[i]:xx[i+1]] for key,value in dataset[n_class][n_name][branch].items()}
                    del dataset[n_class][n_name][branch]
    return(dataset)



def load_dataset_from_pickle(dataset_path):
    with open(dataset_path,"rb") as f:
        dataset = pickle.load(f)
    return(dataset)

if __name__ == "__main__":
    import time

    t0 = time.time()
    dataset = load_dataset_from_scattered_dirs(r"Y:\0000NeuronsData\clean_movies_1_min")        


    pickle_path = r"Y:\0000NeuronsData\clean_movies_1_min\branch_tracking\all_branches_length.pickle"
    t1 = time.time()
    with open(pickle_path,'wb') as f:
        pickle.dump(dataset,f) 
    t2 = time.time()

    dataset2 = load_dataset_from_pickle(pickle_path)
    t3 = time.time()
    print(t1-t0,t2-t1,t3-t2)
