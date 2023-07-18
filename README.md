# Graph Attention Convolutional Neural network (GAT) applied on the NK cell data set

This repository shows my implementation of a GAT applied to the well-discussed NK cell data set. 
Details regarding the dataset and the Download link can be found [here](https://zenodo.org/record/6780417). The dataset consists of 20 .fcs files each containing a single cell flow cytometry measurement and a discreet label of responder, non-responder. Typically for a flow cytometry measurement a very large number of data points are measured ~ $10^4$ data samples, but only on a small sample number.

The general setup is the following: 

The entire dataset $M$ consists of 20 .fsc files. A single file $f_i$ contains ~ $10^4$ cells with a cell $c_j$ being described by 37 proteomics markers. Thus:

$M = \bigcup_{i} f_i = \bigcup_{i,j} c_{i,j}$ 

To craft a reliable data set size from each $f_i$ a total of $s_{sub}$ subsamples are taken, each containing $n$ cells. To craft the graph structure the K-nearest neighborhood graph is applied to the individual subsamples and the edge features are either the inverse of the Euclidian distance or the cosine similarity. To ensure that the train and test data sets are still balanced per patient a fixed number of sub-samples (s_sub) is taken from each patient. This approach can lead to individual cells being in multiple train/test samples. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
the code is designed to be run on multiple GPUs if it needs to be run on a CPU-only machine, 
please change from the torch_geometric.loader.DataListLoader to the standard Torch DataLoader 

## Training

To train the model, please download the data from [here](https://zenodo.org/record/6780417). 

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
- The adjacency feature matrix $\frac{1}{\text{euclidian}}$ 
- The adjacency feature matrix (cosine sim)
- The Label

Finally simply run the hyperparameter search:

```
python hyper_search.py
```

