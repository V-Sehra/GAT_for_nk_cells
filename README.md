# Graph Attention Convolutional Neural network (GAT) applied on the NK cell data set

This repository shows my implementation of a GAT applied to the well-discussed NK cell data set. 
Details regarding the dataset and the Download link can be found [here](https://zenodo.org/record/6780417). /

The general setup is the following: /

there consists 13 Patients from which single cell flow cytometry data were measured. Thus a very large per-patient data set (~10**3 data samples. To craft this into a (geometric) deep learning-compatible training data set from each patient n_cells are used to craft a single data sample. The total number of cells per patient strongly varies. To ensure that the train and test data sets are still balanced per patient a fixed number of sub-samples (s_sub) is taken from each patient. This approach can lead to individual cells being in multiple train/test samples. /

To further create a graph structure on the samples a KNN graph is applied on each sample individually with the edge feature being either the inverse of the euclidian distance or the cosine similarity.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
the code is designed to be run on multiple GPUs if it needs to be run on a CPU-only machine, 
please change from the torch_geometric.loader.DataListLoader to the standard Torch DataLoader 

## Training

To train the model, please download the data from [here](https://zenodo.org/record/6780417). /

Once downloaded run 
```
python train_test_split.py -path_to_marker_csv PATH_TO_FCS_files -path_to_raw_data PATH_TO_MARKER_LIST
```
The script will create a test and train split for the training, but only on the .fcs level. 
To create the trainable graph dataset please run the following:

```
python graph_creator.py -typ TEST_or_TRAIN
```
Now there will be a folder containing all graphs with each graph containing:
- The cell matrix
- The adjacency matrix
- The adjacency feature matrix (1/euclidian)
- The adjacency feature matrix (cosine sim)
- The Label

Finally simple run the hyper parameter search:

```
python hyper_search.py
```

