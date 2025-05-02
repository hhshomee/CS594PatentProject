
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import pandas as pd
import os
import time
class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):

        self.model_name = model_name
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
        total_time = time.time() - start_time
        return embeddings,total_time

    def save_embeddings(self, embeddings, file_path):

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, embeddings)
        print(f" Embeddings saved to {file_path}")

    def load_embeddings(self, file_path):

        embeddings = np.load(file_path)
        print(f" Embeddings loaded from {file_path}")
        return embeddings

    def generate_from_csv(self, csv_path, title_column, abstract_column, batch_size=32, save_path=None):

        df = pd.read_csv(csv_path)
        if title_column not in df.columns or abstract_column not in df.columns:
            raise ValueError(f"Columns '{title_column}' or '{abstract_column}' not found in {csv_path}. Available columns: {df.columns.tolist()}")


        texts = (df[title_column].fillna('') + ' ' + df[abstract_column].fillna('')).tolist()
        print(f" Loaded {len(texts)} concatenated texts from '{title_column} + {abstract_column}'")

        embeddings,total_time = self.generate_embeddings(texts, batch_size=batch_size)

        if save_path is not None:
            self.save_embeddings(embeddings, save_path)

        return embeddings,total_time


# - "all-MiniLM-L6-v2"
# - "distilbert-base-nli-stsb-mean-tokens"
# - "bert-base-nli-mean-tokens"
# - "paraphrase-MiniLM-L12-v2"


model_name = 'all-MiniLM-L6-v2'
embed_gen = EmbeddingGenerator(model_name)



root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
csv_path = os.path.join(root, "data", "data_A61.csv")
# csv_path = "data/data_A61.csv"

save_path = os.path.join(root, "output", "embeddings", "MiniLML6.npy")
title_column = "title"
abstract_column = "abstract"

embeddings,total_time = embed_gen.generate_from_csv(
    csv_path=csv_path,
    title_column=title_column,
    abstract_column=abstract_column,
    batch_size=32,
    save_path=save_path
)

print("Embedding generation complete. Shape:", embeddings.shape)
print(f"Total Time for Embedding: {total_time:.2f} seconds")