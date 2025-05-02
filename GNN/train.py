import time
from torch_geometric.nn import GINConv
import torch.nn.functional as F
import torch.nn as nn
from GNN.gnns import GCN, GCN_deep, GraphSAGE, GraphSAGE_deep, GAT, GAT_deep, GIN, GIN_deep
 
from GNN.data_loader import load_patent_graph


patent_data, texts = load_patent_graph()

model = GCN(patent_data.num_node_features, 64, len(torch.unique(patent_data.y)))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
start_time = time.time()

for epoch in range(50):
    start_epoch = time.time()
    model.train()
    optimizer.zero_grad()
    out = model(patent_data)
    loss = F.cross_entropy(out[patent_data.train_mask], patent_data.y[patent_data.train_mask])
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    epoch_time = time.time() - start_epoch


model.eval()
preds = model(patent_data).argmax(dim=1)
correct = (preds[patent_data.test_mask] == patent_data.y[patent_data.test_mask]).sum()
acc = int(correct) / int(patent_data.test_mask.sum())
print(f"Test accuracy {acc:.2f}")
total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f} seconds")