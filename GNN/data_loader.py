import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def load_patent_graph():
    
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

    
    embedding_path = os.path.join(root, "output", "embeddings", "MiniLML6.npy")
    csv_path = os.path.join(root, "data", "data_A61.csv")
    edge_path = os.path.join(root, "output", "patent_edges.csv")

   
    x = torch.tensor(np.load(embedding_path), dtype=torch.float32)

  
    df = pd.read_csv(csv_path)
    patent_ids = df['id'].tolist()
    id_map = {pid: idx for idx, pid in enumerate(patent_ids)}
    label_map = {l: i for i, l in enumerate(sorted(df['cpc_subclass'].unique()))}
    y = torch.tensor([label_map[l] for l in df['cpc_subclass']], dtype=torch.long)


    edge_df = pd.read_csv(edge_path)
    edge_index = torch.tensor(
        [[id_map[src], id_map[dst]] for src, dst in zip(edge_df['source_id'], edge_df['target_id'])],
        dtype=torch.long
    ).T

    
    N = len(y)
    ids = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(ids)

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[ids[:int(0.6 * N)]] = True
    val_mask[ids[int(0.6 * N):int(0.8 * N)]] = True
    test_mask[ids[int(0.8 * N):]] = True


    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    texts = df['title'] + ' ' + df['abstract']
    return data, texts


patent_data, texts = load_patent_graph()
print(patent_data)
print("Sample text:", texts.iloc[0])
