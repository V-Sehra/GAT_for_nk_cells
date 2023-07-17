#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:18:15 2022

@author: Vivek
"""
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch_geometric.nn import DataParallel
import torch.nn as nn
from torch_geometric.loader import DataListLoader
import os
cwd = os.getcwd()
import data_class as dat_class
import multiprocessing as mp
import json
import argparse
import torch
from sklearn.metrics import balanced_accuracy_score
import model_utils

torch.multiprocessing.set_start_method('fork')
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
# data spezific
parser.add_argument("-num_nei", "--neiborhood_number", type=int, default=90)
parser.add_argument("-n_cells", "--n_cells", type=int, default=1000)
parser.add_argument("-sub_set", "--sub_set", type=int, default=100)
parser.add_argument('-s_m', '--similarity_measure', type=str, default='euclide')
parser.add_argument('-save_model_bool', '--save_model_bool', type=bool, default=False)

#Paths
parser.add_argument('-marker_csv_path', '--marker_csv_path', type=str, default=None)
parser.add_argument('-path_data', '--path_data', type=str, default=None)
parser.add_argument('-save_path', '--save_path', type=str, default=None)


# training spezific
parser.add_argument("-input_dim", "--input_dim", type=int, default=37)
parser.add_argument("-epochs", "--epochs", type=int, default=10)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2)
parser.add_argument("-final_layer", "--final_layer", type=int, default=2)
parser.add_argument("-batch_size", "--batch_size", type=int, default=85)
parser.add_argument("-repeat_num_training", "--repeat_num_training", type=int, default=5)


args = parser.parse_args()
print(args)

n_cells = args.n_cells
num_nei = args.neiborhood_number
sub_set = args.sub_set
save_model_bool = args.save_model_bool

#to ensure the hyperparameter performance was statistical outlier repeat the number of trainings
repeat_num_training = args.repeat_num_training

input_dim = args.input_dim
similarity_measure = args.similarity_measure

## paths for data and model saving
path_data = args.path_data
if path_data is None:
    path_data = os.path.join(f'{cwd}', 'data')

marker_csv_path = args.marker_csv_path
if marker_csv_path is None:
    path_marker = os.path.join(f'{cwd}','data','NK_markers.csv')
else:
    path_marker = marker_csv_path

save_path = args.save_path
if save_path is None:
    save_path = os.path.join(f'{cwd}', 'model',f'{similarity_measure}',f'{n_cells}',f'num_nei_{num_nei}')
else:
    save_path = save_path
os.makedirs(save_path, exist_ok=True)


## training spesific information
epochs = args.epochs
final_layer = args.final_layer
batch_size = args.batch_size
learning_rate = args.learning_rate


data_path_train = os.path.join(f'{path_data}','train','graphs',f'{n_cells}',f'num_nei_{num_nei}')
os.makedirs(data_path_train, exist_ok=True)

data_path_test = os.path.join(f'{path_data}','test','graphs',f'{n_cells}',f'num_nei_{num_nei}')
os.makedirs(data_path_test, exist_ok=True)


total_workers = mp.cpu_count()


data_train = DataListLoader(
    dat_class.nk_graph_att_data(root=data_path_train, path = path_data,
                      num_nei=num_nei,sample_number=sub_set, n_cells=n_cells,
                  typ='train',marker_csv_path = path_marker),
    batch_size=batch_size, shuffle=True, num_workers=5,
    prefetch_factor=2)

data_test = DataListLoader(
    dat_class.nk_graph_att_data(root=data_path_test, path = path_data,
                      num_nei=num_nei,sample_number=sub_set, n_cells=n_cells,
                  typ='test',marker_csv_path = path_marker),
    batch_size=batch_size, shuffle=True, num_workers=5,
    prefetch_factor=2)

# Balance the Loss funktion as the dataset is very unbalanced
class_weights = []
obs = pd.read_csv(os.path.join(f'{path_data}', 'cell_train.csv'))
class_weights.append(1 - len(obs['label'][obs['label'] == 0]) / len(obs))
class_weights.append(1 - len(obs['label'][obs['label'] != 0]) / len(obs))
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))

# Previous hyperparameter searches within this data type showed an encoder structure to be useful
# Thus the three Layer with end bottleneck structe will be focused

feat_f1_vec = np.arange(input_dim, 5, -5)
dp_vec = np.arange(0.1, 1, 0.3)

for f_1 in feat_f1_vec:
    feat_f2_vec = np.arange(f_1, 5, -5)

    for f_2 in feat_f2_vec:
        feat_f3_vec = np.arange(f_1,f_2, -5)
        for f_3 in feat_f3_vec:
            for dp in dp_vec:

                model_name = f"f1_{f_1}_f2_{f_2}_f3_{f_3}_dp_{dp}".replace('.','_')


                #create a JSON file containing all relavant training information from this architekture
                model_utils.create_meta_file(save_path_meta_file = os.path.join(f'{save_path}', f'{model_name}_meta_info.json'),
                                            dropout_rate = dp,
                                            dimension_per_layer_vec = [f_1,f_2,f_3],
                                            epochs = epochs,
                                             learning_rate = learning_rate)

                #collect the train_loss, vall_loss, val_acc and bal_acc from all models with the same architekture
                performance = {}


                for num in tqdm(range(repeat_num_training)):

                    perf = []

                    model = model_utils.att_GNN(num_of_feat=input_dim,
                                             f_1=int(f_1),
                                             f_2=int(f_2),
                                             f_3=int(f_3),
                                             dp=dp,
                                             f_final=2, Pre_norm=True,
                                            similarity_typ=similarity_measure)


                    model = DataParallel(model)

                    model.to(device)


                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    for ep in tqdm(range(epochs)):

                        optimizer.zero_grad()
                        for train_sample in data_train:
                            optimizer.zero_grad()
                            output = model(train_sample)

                            y = torch.tensor([data.y for data in train_sample]).to(output.device)

                            loss = criterion(output, y)
                            loss.backward()
                            optimizer.step()
                    #model = model_utils.train_loop(model = model,
                    #                               optimizer = optimizer,
                    #                               epochs = epochs,
                    #                               data_list_set = data_train,
                    #                               loss_fkt = criterion)
                    if save_model_bool:
                        torch.save(model.state_dict(), os.path.join(f'{save_path}', f'{model_name}_id_{num}.pt'))

                    model.eval()

                    perf = []

                    with torch.no_grad():
                        for test_sample in data_test:
                            pred = model(test_sample)

                            target_test = torch.tensor([data.y for data in test_sample]).to(pred.device)

                            vall_loss = criterion(pred, target_test).item()

                            _, value_pred = torch.max(pred, dim=1)

                            acc = ((value_pred == target_test).sum() / len(target_test)).item()
                            balanced_accuracy = balanced_accuracy_score(value_pred.cpu(), target_test.cpu())

                            perf.append([acc, vall_loss])


                        performance[str(num)] = perf

                with open(f"{save_path}{model_name}_performance.json", 'w') as outfile:
                    json.dump(performance, outfile)
