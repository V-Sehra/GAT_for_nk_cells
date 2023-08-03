#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm
from tqdm import tqdm
import json


class att_GNN(nn.Module):
    def __init__(self, num_of_feat, f_1, f_2, f_3,
                 dp, Pre_norm, f_final=2, edge_dim = 1,similarity_typ = 'euclid'):
        super(att_GNN, self).__init__()
        self.conv1 = GATv2Conv(num_of_feat, f_1,edge_dim=edge_dim)
        self.conv2 = GATv2Conv(f_1, f_2,edge_dim=edge_dim)
        self.conv3 = GATv2Conv(f_2, f_3,edge_dim=edge_dim)
        self.lin = Linear(f_3, f_final)
        self.dp = dp
        self.Pre_norm = Pre_norm
        self.BatchNorm = BatchNorm(num_of_feat)
        self.similarity_typ = similarity_typ

    def forward(self, data_input):
        x = data_input.x.float()
        edge_index = data_input.edge_index
        if self.similarity_typ == 'euclid' or self.similarity_typ == 'euclide':
            edge_att = data_input.euclid
        elif self.similarity_typ == 'cosine':
            edge_att = data_input.cosine

        if self.Pre_norm:
            x = self.BatchNorm(x)

        x = self.conv1(x=x,
                        edge_index=edge_index,
                        edge_attr = edge_att)
        x = F.relu(x)
        x = F.dropout(x, p=self.dp, training=self.training)

        x = self.conv2(x=x,
                        edge_index=edge_index,
                        edge_attr = edge_att)
        x = F.relu(x)
        x = F.dropout(x, p=self.dp, training=self.training)

        x = self.conv3(x=x,
                        edge_index=edge_index,
                        edge_attr = edge_att)
        x = F.relu(x)
        x = F.dropout(x, p=self.dp, training=self.training)

        x = global_mean_pool(x, data_input.batch)
        x = self.lin(x)

        return F.softmax(x, dim=1)



def train_loop(model,optimizer,epochs,data_list_set,loss_fkt):
    model.train()
    for ep in tqdm(range(epochs)):

        optimizer.zero_grad()
        for train_sample in data_list_set:
            optimizer.zero_grad()
            output = model(train_sample)

            y = torch.tensor([data.y for data in train_sample]).to(output.device)

            loss = loss_fkt(output, y)
            loss.backward()
            optimizer.step()
    return(model)


def create_meta_file(dropout_rate, dimension_per_layer_vec, epochs, learning_rate, save_path_meta_file):
    meta_json = {}

    meta_json['dp'] = f'{dropout_rate}'
    meta_json['epochs'] = f'{epochs}'
    meta_json['lr'] = f'{learning_rate}'

    for feature_layer_idx in range(len(dimension_per_layer_vec)):
        meta_json[f'feature_layer_{feature_layer_idx}'] = f'{dimension_per_layer_vec[feature_layer_idx]}'

    with open(save_path_meta_file, 'w') as outfile:
        json.dump(meta_json, outfile)