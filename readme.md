# Text-Attributed-Graph on Patent Data

This repository contains code to build a text-attributed graph from patent data. The graph represents patents as nodes and connects them based on semantic similarity of their textual content.

## Features

- Load and preprocess patent data from CSV files
- Generate semantic embeddings using SentenceTransformers
- Build a similarity graph based on customizable thresholds

## Patent Graph Dataset
- is located in `CS594PatentProject/output` folder
  
## Generate Text Embeddings
- Go to `GNN/embedding.py`
- Set the model you want to use (e.g., `all-MiniLM-L6-v2`)
- Embeddings will be saved to: output/embeddings/MiniLML6.npy

## Data Loader
- Go to `GNN/data_loader.py`
- Set the appropriate embedding file:

embedding_path = os.path.join(root, "output", "embeddings", "MiniLML6.npy")

## Train the GNN Model

To train a GNN on the patent graph

- Open `GNN/train.py`
- Choose the GNN model by modifying the line:
model = GCN(patent_data.num_node_features, 64, len(torch.unique(patent_data.y))) 
Replace GCN with the specific GNN model you want to use, such as GCN_deep, GraphSAGE, GAT, GIN, etc.

## Generate explanation

- Open TAPE_modified and run explanation.py for explanation generation
- Put your huggingface access token in the line:
hf_token = "hf_token"

## Authors
Homaira huda Shomee (hshome2@uic.edu)
Ataher Sams (asams3@uic.edu)
<!-- - Save graphs in various formats (GraphML, GEXF, Pickle, CSV)


