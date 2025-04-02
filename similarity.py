import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. PyG export functionality will not be available.")

class SimilarityGraph:
    def __init__(self):
        
        self.graph = None
        self.similarity_matrix = None
        self.embeddings = None
        self.patent_ids = None
        
    def compute_similarity_matrix(self, embeddings, metric='cosine'):
        
        distances = pdist(embeddings, metric=metric)
        dist_matrix = squareform(distances)
        
        if metric == 'cosine':
            similarity_matrix = 1 - dist_matrix
        
        self.similarity_matrix = similarity_matrix
        self.embeddings = embeddings
        return similarity_matrix
    
    def build_graph(self, similarity_matrix, patent_ids, threshold=0.7):
        
        n = similarity_matrix.shape[0]
        G = nx.Graph()
        
        for i in range(n):
            G.add_node(i, patent_id=patent_ids[i])
        
        # Add edges with similarity scores above threshold
        edges = []
        for i in tqdm(range(n)):
            for j in range(i+1, n):
                if similarity_matrix[i, j] >= threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
                    edges.append((i, j, similarity_matrix[i, j]))
        
        print(f"Graph built with {n} nodes and {len(edges)} edges (threshold={threshold})")
        self.graph = G
        self.patent_ids = patent_ids
        return G
    
    def get_connected_components(self):
        """
        Get connected components of the graph
        """
        if self.graph is None:
            raise ValueError("Graph not built yet")
            
        return list(nx.connected_components(self.graph))
    
    def save_graph(self, file_path, format='graphml'):
        """
        Save the graph to a file
        """
        if self.graph is None:
            raise ValueError("Graph not built yet")
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if format == 'graphml':
            nx.write_graphml(self.graph, file_path)
        elif format == 'gexf':
            nx.write_gexf(self.graph, file_path)
        elif format == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(self.graph, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"Graph saved to {file_path}")
    
    def load_graph(self, file_path, format='graphml'):
        """
        Load the graph from a file
        """
        if format == 'graphml':
            self.graph = nx.read_graphml(file_path)
        elif format == 'gexf':
            self.graph = nx.read_gexf(file_path)
        elif format == 'pickle':
            with open(file_path, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"Graph loaded from {file_path}")
        return self.graph
    
    def visualize_graph(self, output_file=None, figsize=(12, 12)):
        """
        Visualize the graph
        """
        if self.graph is None:
            raise ValueError("Graph not built yet")
            
        plt.figure(figsize=figsize)
        
        
        pos = nx.spring_layout(self.graph)
        
       
        nx.draw_networkx_nodes(self.graph, pos, node_size=50, alpha=0.8)
        
        
        edge_weights = [self.graph[u][v]['weight'] * 2 for u, v in self.graph.edges()]
        nx.draw_networkx_edges(self.graph, pos, width=edge_weights, alpha=0.5)
        
       
        
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to {output_file}")
        
        plt.show()
        
    def save_edge_csv(self, file_path, patent_data=None):
        """
        Save edge list with all available information as CSV
        
        Args:
            file_path: Path to save CSV file
            patent_data: Optional DataFrame containing additional patent information
        """
        if self.graph is None:
            raise ValueError("Graph not built yet")
            
        
        edges = []
        for u, v, data in self.graph.edges(data=True):
            source_id = self.graph.nodes[u].get('patent_id', u)
            target_id = self.graph.nodes[v].get('patent_id', v)
            
            edge = {
                'source': u,
                'target': v,
                'source_id': source_id,
                'target_id': target_id,
                'weight': data['weight']
            }
            edges.append(edge)
            
        
        edge_df = pd.DataFrame(edges)
        
       
        if patent_data is not None:
            # Create a mapping from patent_id to row index
            id_col = [col for col in patent_data.columns if 'id' in col.lower()][0]
            patent_id_map = dict(zip(patent_data[id_col], patent_data.index))
            
           
            for col in patent_data.columns:
                if col != id_col:
                    edge_df[f'source_{col}'] = edge_df['source_id'].map(
                        lambda x: patent_data.loc[patent_id_map.get(x, -1), col] 
                        if x in patent_id_map else None
                    )
                    
                    edge_df[f'target_{col}'] = edge_df['target_id'].map(
                        lambda x: patent_data.loc[patent_id_map.get(x, -1), col] 
                        if x in patent_id_map else None
                    )
        
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        edge_df.to_csv(file_path, index=False)
        print(f"Edge list saved to {file_path}")
        
        return edge_df
        
    def save_pytorch_graph(self, file_path, node_features=None):
        """
        Save graph in PyTorch format for use with Graph Neural Networks
        
        Args:
            file_path: Path to save the PyTorch graph
            node_features: Optional node feature matrix. If None, embeddings will be used.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
            
        if self.graph is None:
            raise ValueError("Graph not built yet")
            
       
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
       
        num_nodes = self.graph.number_of_nodes()
        
        
        node_idx = {node: i for i, node in enumerate(self.graph.nodes())}
        
       
        edge_index = []
        edge_attr = []
        
        for u, v, data in self.graph.edges(data=True):
           
            edge_index.append([node_idx[u], node_idx[v]])
            edge_index.append([node_idx[v], node_idx[u]])  
            
           
            edge_attr.append(data['weight'])
            edge_attr.append(data['weight'])  
            
        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()  
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).reshape(-1, 1) 
        
        
        if node_features is not None:
            x = torch.tensor(node_features, dtype=torch.float)
        elif self.embeddings is not None:
            x = torch.tensor(self.embeddings, dtype=torch.float)
        else:
            # If no features available, create identity features
            x = torch.eye(num_nodes, dtype=torch.float)
            
        
        patent_ids = {}
        for node, data in self.graph.nodes(data=True):
            patent_ids[node_idx[node]] = data.get('patent_id', node)
            
        
        graph_data = {
            'x': x, 
            'edge_index': edge_index,  
            'edge_attr': edge_attr,  
            'num_nodes': num_nodes,
            'patent_ids': patent_ids
        }
        
       
        torch.save(graph_data, file_path)
        print(f"Graph saved in PyTorch format to {file_path}")
        
        return graph_data