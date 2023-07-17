#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:43:33 2023

@author: Vivek
"""

import os
cwd = os.getcwd()
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import torch
import multiprocessing as mp
import functools
import argparse
import utilities

# receive the argument of train or test
parser = argparse.ArgumentParser()
parser.add_argument("-typ", "--typ", type=str, default='test')
parser.add_argument("-n_cells", "--n_cells", type=int, default=1000)
parser.add_argument("-num_nei", "--neiborhood_number", type=int, default=90)
parser.add_argument("-sub_set", "--sub_set", type=int, default=100)
parser.add_argument("-marker_csv_path", "--marker_csv_path", type=str, default=None)
parser.add_argument("-path_data", "--path_data", type=str, default=None)

args = parser.parse_args()
print(args)

#specs regarding the sampling from the measurment in respect to the final graph
num_nei = args.neiborhood_number
typ = args.typ
n_cells = args.n_cells
sub_set = args.sub_set

marker_csv_path = args.marker_csv_path

if marker_csv_path == None:
    marker_csv_path = os.path.join(f'{cwd}','data','NK_markers.csv')
else:
    marker_csv_path = marker_csv_path

path_data = args.path_data
if path_data == None:
    path_save = os.path.join(f'{cwd}','data')
else:
    path_save = path_data

cosine = torch.nn.CosineSimilarity(dim=1)


def batch_data(batch, mat, neibhour_number, save_path, label_matrix):

    for pat_id in range(len(mat)):
        if not os.path.exists(f'{save_path}/graph_pat_{pat_id}_{batch}.pt'):
            cells = torch.tensor(mat[pat_id][batch])

            edge,euc = utilities.calc_edge_mat(cells, num_neibours=neibhour_number, dev='cpu',dist_bool = True)
            edge, euc = utilities.remove_zeros_edge_mat(edge, euc)

            cosine_sim = cosine(cells[edge[0]],cells[edge[1]])

            data = Data(x=cells,
                        edge_index=torch.tensor(edge).long(),
                        euclid = torch.tensor(1/euc),
                        cosine = cosine_sim,
                        y=torch.tensor(np.array([label_matrix[pat_id]])).flatten())
            torch.save(data, os.path.join(f'{save_path}', f'graph_pat_{pat_id}_{batch}.pt'))

    return


# load the csv file containing the patient ids
obs = pd.read_csv(os.path.join(f'{path_save}',f'cell_{typ}.csv'), low_memory=False)


files = obs['fcs_filename'].unique()
# load the train cell values
marker_idx = 0
pat_mat = []
label_mat = []

fcs_file_path = os.path.join(f'{path_save}',f'{typ}_raw_files')


# subsample over the patients to create the 100 files
for file in files:
    label_mat.append(obs['label'][obs['fcs_filename'] == file].iloc[0])

    path_file = os.path.join(f'{fcs_file_path}', f'{file}')
    marker_idx = utilities.get_marker_idx(path_file, marker_csv_path)
    pat_mat.append(utilities.sub_sampler(input_data = utilities.load_single_fcs(path = path_file,
                                                           idx = marker_idx,
                                                           normalize_bool = False)[0],
                                        sub_set = sub_set,
                                        n_cells = n_cells))

batch = np.arange(0, 100)
pool = mp.Pool(mp.cpu_count()-int(mp.cpu_count()*0.8))


save_path_graph = os.path.join(f'{path_save}',f'{typ}','graphs',f'{n_cells}',f'num_nei_{num_nei}')
os.makedirs(save_path_graph, exist_ok=True)

results = pool.map(
    functools.partial(batch_data,
                      mat=pat_mat,
                      label_matrix=label_mat,
                      neibhour_number=num_nei,
                      save_path=save_path_graph), batch)

pool.close()
