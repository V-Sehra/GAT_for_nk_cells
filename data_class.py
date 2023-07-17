from torch_geometric.data import Dataset, Data
import pandas as pd
import os
import utilities
import numpy as np
import torch
cosine = torch.nn.CosineSimilarity(dim=1)
cwd = os.getcwd()

class nk_graph_att_data(Dataset):
    def __init__(self, root, path,marker_csv_path,
                 num_nei=20,
                 n_cells=500, typ='train',
                 sample_number=100):

        self.path = path
        self.n_cells = n_cells
        self.typ = typ
        self.num_nei = num_nei
        self.sample_number = sample_number
        self.obs = None
        self.marker_csv_path = marker_csv_path
        self.marker_idx = None

        super().__init__(root)

    @property
    def processed_dir(self):
        return self.root

    @property
    def processed_file_names(self):

        if self.obs is None:
            self.obs = pd.read_csv(os.path.join(f'{self.path}',f'cell_{self.typ}.csv'), low_memory=False)
        pat_number = len(self.obs['fcs_filename'])

        graph_file_names = [f'graph_pat_{pat_counter}_{sample_counter}.pt' for pat_counter in range(pat_number)
                            for sample_counter in range(self.sample_number)]

        return graph_file_names

    def process(self):
        idx = 0
        print('processing')
        if self.obs is None:
            self.obs = pd.read_csv(os.path.join(f'{self.path}',f'cell_{self.typ}.csv'), low_memory=False)


        files = self.obs['fcs_filename'].unique()
        # load the train cell values

        pat_mat = []
        label_mat = []
        fcs_file_path = f"{self.path}/{self.typ}_raw_files/"

        # subsample over the patients to create the 100 files
        for file in files:

            path_file = f"{fcs_file_path}/{file}"
            marker_idx = utilities.get_marker_idx(path_file, self.marker_csv_path)

            label_mat.append(self.obs['label'][self.obs['fcs_filename'] == file].iloc[0])

            pat_mat.append(utilities.sub_sampler(input_data = utilities.load_single_fcs(path =path_file,
                                                                                        idx = marker_idx,
                                                                                        normalize_bool = False)[0],
                                                 sub_set = self.sample_number,
                                                 n_cells = self.n_cells))

        for batch in range(self.sample_number):
            for pat_id in range(len(pat_mat)):
                cells = torch.tensor(pat_mat[pat_id][batch])
                edge,euc = utilities.calc_edge_mat(cells,
                                                      num_neibours=self.num_nei,
                                                      dev='cpu',dist_bool = True)
                edge, edge_att = utilities.remove_zeros_edge_mat(edge, euc)

                cosine_sim = cosine(cells[edge[0]], cells[edge[1]])


                data = Data(x=torch.tensor(cells),
                            edge_index=torch.tensor(edge).long(),
                            euclid=torch.tensor(1 / euc),
                            cosine = cosine_sim,
                            y=torch.tensor(np.array([label_mat[pat_id]])).flatten())
                torch.save(data, f'{self.processed_dir}/graph_pat_{pat_id}_{batch}.pt')
                idx += 1


        return

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):

        data = torch.load(f'{self.processed_dir}/{self.processed_file_names[idx]}')
        return data