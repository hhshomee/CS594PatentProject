
import os
import time
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from ..GNN.gnns import GCN, GCN_deep, GraphSAGE, GraphSAGE_deep, GAT, GAT_deep, GIN, GIN_deep
 

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(root, "data", "data_A61.csv")
explanation_csv = os.path.join(root, "output", "A61_explanation.csv")
embedding_path = os.path.join(root, "output", "embeddings", "embeddings.npy")
explanation_embedding_path = os.path.join(root, "output", "embeddings", "explanation_embeddings.npy")
edges_path = os.path.join(root, "output", "patent_edges.csv")

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f" Model '{model_name}' loaded on {self.device}")

    def generate_embeddings(self, texts, batch_size=32, show_progress_bar=True):
        start_time = time.time()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        return embeddings, time.time() - start_time

    def save_embeddings(self, embeddings, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, embeddings)
        print(f"Saved embeddings to {file_path}")

    def generate_from_csv(self, csv_path, explanation_column, batch_size=32, save_path=None):
        df = pd.read_csv(csv_path)
        texts = df[explanation_column].fillna('').tolist()
        print(f" Loaded {len(texts)} explanation texts")
        embeddings, total_time = self.generate_embeddings(texts, batch_size=batch_size)
        if save_path:
            self.save_embeddings(embeddings, save_path)
        return embeddings, total_time


embed_gen = EmbeddingGenerator()
_, total_time = embed_gen.generate_from_csv(
    csv_path=explanation_csv,
    explanation_column="explanation",
    save_path=explanation_embedding_path
)
print("Explanation embeddings generated.")
print(f" Time taken: {total_time:.2f} seconds")


def load_merged_patent_graph():
    df = pd.read_csv(csv_path).reset_index(drop=True)
    patent_ids = df['id'].tolist()
    id_map = {pid: idx for idx, pid in enumerate(patent_ids)}
    label_map = {l: i for i, l in enumerate(sorted(df['cpc_subclass'].unique()))}
    y = torch.tensor([label_map[l] for l in df['cpc_subclass']], dtype=torch.long)

    edge_df = pd.read_csv(edges_path)
    edge_df = edge_df[edge_df['source_id'].isin(patent_ids) & edge_df['target_id'].isin(patent_ids)]
    edge_index = torch.tensor(
        [[id_map[src], id_map[dst]] for src, dst in zip(edge_df['source_id'], edge_df['target_id'])],
        dtype=torch.long
    ).T

    h_orig_all = np.load(embedding_path)
    h_expl = np.load(explanation_embedding_path)
    full_df = pd.read_csv(csv_path)
    id_to_row = {pid: idx for idx, pid in enumerate(full_df['id'].tolist())}
    h_orig_sub = np.stack([h_orig_all[id_to_row[pid]] for pid in patent_ids])

    assert h_orig_sub.shape[0] == h_expl.shape[0], "Shape mismatch between original and explanation embeddings"

    x = torch.tensor(np.concatenate([h_orig_sub, h_expl], axis=1), dtype=torch.float32)

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

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

patent_data = load_merged_patent_graph()
print("Graph data loaded:", patent_data)



model = GCN(patent_data.num_node_features, 64, len(torch.unique(patent_data.y)))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 50
start_time = time.time()

print("Training started...")
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(patent_data)
    loss = F.cross_entropy(out[patent_data.train_mask], patent_data.y[patent_data.train_mask])
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    logits = model(patent_data)
    preds = logits.argmax(dim=1)
    correct = (preds[patent_data.test_mask] == patent_data.y[patent_data.test_mask]).sum()
    test_acc = correct.item() / patent_data.test_mask.sum().item()

print(f"Final Test Accuracy: {test_acc:.4f}")
print(f" Total Training Time: {time.time() - start_time:.2f} seconds")
