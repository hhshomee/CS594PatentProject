from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import os

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
       
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
    def generate_embeddings(self, texts, batch_size=32, show_progress_bar=True):
        
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        return embeddings
    
    def save_embeddings(self, embeddings, file_path):
        """
        Save embeddings to a numpy file
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, embeddings)
        print(f"Embeddings saved to {file_path}")
        
    def load_embeddings(self, file_path):
        """
        Load embeddings from a numpy file
        """
        embeddings = np.load(file_path)
        print(f"Embeddings loaded from {file_path}")
        return embeddings