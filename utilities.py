
import flowio
import numpy as np
import random as rnd
import faiss
import pandas as pd
import os
import shutil


class FcmData(object):
    def __init__(self, events, channels):
        self.channels = channels
        self.events = events
        self.shape = events.shape

    def __array__(self):
        return self.events

def loadFCS(filename, *args, **kwargs):
    f = flowio.FlowData(filename, ignore_offset_error=True)
    events = np.reshape(f.events, (-1, f.channel_count))
    channels = []
    for i in range(1, f.channel_count + 1):
        key = str(i)
        if 'PnS' in f.channels[key] and f.channels[key]['PnS'] != u' ':
            channels.append(f.channels[key]['PnS'])
        elif 'PnN' in f.channels[key] and f.channels[key]['PnN'] != u' ':
            channels.append(f.channels[key]['PnN'])
        else:
            channels.append('None')
    return FcmData(events, channels)

def get_marker_idx(fcs_file, path_marker_csv):
    if fcs_file[-4:] != '.fcs':
        fcs_file = fcs_file+'.fcs'

    markers = pd.read_csv(path_marker_csv).columns

    fcs_mat = loadFCS(fcs_file)
    marker_idx = [fcs_mat.channels.index(label) for label in markers]

    return (marker_idx)

def sub_sampler(input_data, sub_set, n_cells):
    subsampled_cell = []

    for s_sub in range(sub_set):
        if len(input_data) < n_cells:

            if n_cells / len(input_data) > 2:
                sub_id = rnd.sample(range(len(input_data)), len(input_data))
                append_data = input_data[sub_id]
                for i in range(1, n_cells // len(input_data)):
                    app_id = rnd.sample(range(len(input_data)), len(input_data))

                    app = input_data[app_id]
                    append_data = np.append(append_data, app, axis=0)

                app_id = rnd.sample(range(len(input_data)), n_cells - len(input_data) * (n_cells // len(input_data)))

                app = input_data[app_id]
                append_data = np.append(append_data, app, axis=0)

                subsampled_cell.append(append_data)


            else:
                sub_id = rnd.sample(range(len(input_data)), len(input_data))

                app_id = rnd.sample(range(len(input_data)), n_cells - len(input_data))

                append_data = input_data[sub_id]
                append_data = np.append(append_data, input_data[app_id], axis=0)

                subsampled_cell.append(append_data)


        else:

            sub_id = rnd.sample(range(len(input_data)), n_cells)
            subsampled_cell.append(input_data[sub_id])

    return (subsampled_cell)

def ftrans(x, c):
    return np.arcsinh(1. / c * x)

def load_single_fcs(path, idx=None,normalize_bool = True):
    data = loadFCS(path).events
    mat = []
    if normalize_bool:
        if idx is not None:
            mat.append(ftrans(np.array(data[:, idx]), 5))
        else:
            mat.append(ftrans(np.array(data), 5))
    else:
        if idx is not None:
            mat.append(np.array(data[:, idx]))
        else:
            mat.append(np.array(data))

    return (mat)

def calc_edge_mat(mat, num_neibours, dev,dist_bool = False):

    x = mat
    index = faiss.IndexFlatL2(np.shape(x)[1])
    x = np.ascontiguousarray(x)
    if dev == 'cpu':

        index.add(x)
        k = int(num_neibours)
        D, I = index.search(x, k)

    else:
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)

        gpu_index_flat.add(x)

        k = int(num_neibours)
        D, I = gpu_index_flat.search(x, k)

    edge = reshape_faiss_to_pyGeo(I)

    if dist_bool == True:
        edge_att = reshape_faiss_to_pyGeo(D, dist_bool=True)
        return(edge,edge_att)
    else:
        return (edge)

def reshape_faiss_to_pyGeo(to_rehape_mat, dist_bool = False):
    if not dist_bool:

        new_shaped_mat = []
        for row in range(len(to_rehape_mat)):
            temp_mat = []

            for col in range(1, len(to_rehape_mat[row])):
                if len(temp_mat) == 0:
                    temp_mat = [to_rehape_mat[row, 0], to_rehape_mat[row, col]]
                else:
                    temp_mat = np.vstack([temp_mat, [to_rehape_mat[row, 0], to_rehape_mat[row, col]]])

            if len(new_shaped_mat) == 0:
                new_shaped_mat = temp_mat
            else:
                new_shaped_mat = np.vstack([new_shaped_mat, temp_mat])

        if np.shape(new_shaped_mat)[0] != 2:
            new_shaped_mat = np.transpose(new_shaped_mat)

    if dist_bool:

        new_shaped_mat = []
        for row in range(len(to_rehape_mat)):

            for col in range(1, len(to_rehape_mat[row])):
                if len(new_shaped_mat) == 0:
                    new_shaped_mat = [to_rehape_mat[row, col]]
                else:
                    new_shaped_mat = np.append(new_shaped_mat, to_rehape_mat[row, col])


    return(new_shaped_mat)

def remove_zeros_edge_mat(edge,dist,limit = 0):
    zero_index = np.where(dist <= limit)[0]
    if len(zero_index) !=0 :
        edge = np.delete(edge, zero_index, 1)
        dist = np.delete(dist, zero_index, 0)
    return(edge,dist)


def file_mover(list_file_names,sorce_folder,dist_folder):

    for file in list_file_names:
        src = os.path.join(f'{sorce_folder}', f'{file}')
        dst = os.path.join(f'{dist_folder}', f'{file}')
        shutil.copyfile(src, dst)