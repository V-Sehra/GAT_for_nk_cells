#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:29:27 2022

@author: Vivek
"""
import os
cwd = os.getcwd()
import pandas as pd
import random as rnd
import utilities
import numpy as np
import argparse

#import all nesseary paths => where is the raw data saved and where should the fcs files to saved
parser = argparse.ArgumentParser()
parser.add_argument("-path_to_raw_data", "--path_to_raw_data", type=str, default=None)
parser.add_argument("-path_to_marker_csv", "--path_to_marker_csv", type=str, default=None)
parser.add_argument("-save_path", "--save_path", type=str, default=None)
parser.add_argument("-split", "--split", type=float, default=0.3)


args = parser.parse_args()
path_to_raw_data = args.path_to_raw_data
path_to_marker_csv = args.path_to_marker_csv
save_path = args.save_path

split = args.split

if path_to_raw_data is None:
    path_data = os.path.join(f'{cwd}','NK_cell_dataset','NK_cell_dataset','gated_NK')
else:
    path_data = path_to_raw_data

if path_to_marker_csv is None:
    csv = pd.read_csv(os.path.join(f'{cwd}','NK_cell_dataset','NK_fcs_samples_with_labels.csv'))
else:
    csv = pd.read_csv(path_to_marker_csv)

if save_path is None:
    csv_save_path =os.path.join(f'{cwd}','data')
else:
    csv_save_path = save_path

csv_save_path_train = os.path.join(f'{csv_save_path}','train')
csv_save_path_test = os.path.join(f'{csv_save_path}','test')

os.makedirs(csv_save_path_train, exist_ok=True)
os.makedirs(csv_save_path_test, exist_ok=True)


test_ids = rnd.sample(range(len(csv)),int(len(csv)*split))
check = csv.iloc[test_ids]['label'].all() or not csv.iloc[test_ids]['label'].any()

if check:
    while not check:
        test_ids = rnd.sample(range(len(csv)),int(len(csv)*split))
        check = csv.iloc[test_ids]['label'].all() or not csv.iloc[test_ids]['label'].any()
        
        

cell_test = csv.iloc[test_ids]
train_id = np.delete(np.arange(0,len(csv)),test_ids)

cell_train = csv.iloc[train_id]


cell_train.to_csv(os.path.join(f'{csv_save_path}','cell_train.csv'))
cell_test.to_csv(os.path.join(f'{csv_save_path}','cell_test.csv'))



utilities.file_mover(list_file_names = cell_train['fcs_filename'],
                     sorce_folder = path_data,
                     dist_folder = csv_save_path_train)

utilities.file_mover(list_file_names = cell_test['fcs_filename'],
                     sorce_folder = path_data,
                     dist_folder = csv_save_path_train)


