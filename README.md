# Graph Attention Convolutional Neural network (GAT) applied on the NK cell data set

This repository shows my implementation of a GAT applied on the well disscused NK cell data set. 
Details regarding the dataset and the Download link can be found [here](https://zenodo.org/record/6780417). /

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
the code is desgined to be run on multiple GPU's if it needs to be run on a CPU only machine, 
please change from the torch_geometric.loader.DataListLoader to the standart Torch DataLoader 

## Training

To train the model please first download the data from [here](https://zenodo.org/record/6780417). /
Once downloaded run 
```
python train_test_split.py -path_to_marker_csv PATH_TO_FCS_files -path_to_raw_data PATH_TO_MARKER_LIST
```
The skript will create a test and train split for the training, but only on the .fcs level. 
To create the trainable graph dataset please run:

```
python graph_creator.py -typ TEST_or_TRAIN
```
Now there will be a folder containing all graphs including the inverse of the euclidian distance between each cell of the KNN graph, aswell as the cosine similarity.

