import os
import time
from data_loader import PatentDataLoader
from embedding import EmbeddingGenerator
from similarity import SimilarityGraph
import config

def main():
   
    start_time = time.time()
    
    
    
    loader = PatentDataLoader(config.CSV_FILE)
    data = loader.prepare_data(combine_fields=True)
    
    embeddings_file = config.EMBEDDINGS_FILE
    
    if os.path.exists(embeddings_file):
        print(f"Loading cached embeddings from {embeddings_file}")
        embeddings_generator = EmbeddingGenerator(model_name=config.MODEL_NAME)
        embeddings = embeddings_generator.load_embeddings(embeddings_file)
    else:
        print(f"Generating new embeddings using model: {config.MODEL_NAME}")
        embeddings_generator = EmbeddingGenerator(model_name=config.MODEL_NAME)
        embeddings = embeddings_generator.generate_embeddings(data['combined_text'].tolist())
        embeddings_generator.save_embeddings(embeddings, embeddings_file)
    
    # Graph building
    
    graph_builder = SimilarityGraph()
    similarity_matrix = graph_builder.compute_similarity_matrix(embeddings, metric=config.METRIC)
    
    
    id_column = data.columns[data.columns.str.contains('id', case=False)][0]
    
    graph = graph_builder.build_graph(
        similarity_matrix, 
        data[id_column].tolist(),
        threshold=config.SIMILARITY_THRESHOLD
    )
    
    
    # Save as GraphML
    graphml_file = os.path.join(config.OUTPUT_DIR, "patent_graph.graphml")
    graph_builder.save_graph(graphml_file, format="graphml")
    
    # Save as Pickle
    pickle_file = os.path.join(config.OUTPUT_DIR, "patent_graph.pkl")
    graph_builder.save_graph(pickle_file, format="pickle")
    
    # Save as CSV
    csv_file = os.path.join(config.OUTPUT_DIR, "patent_edges.csv")
    graph_builder.save_edge_csv(csv_file, patent_data=data)
    
    # Save as PyTorch graph
    pt_file = os.path.join(config.OUTPUT_DIR, "graph.pt")
    try:
        graph_builder.save_pytorch_graph(pt_file, node_features=embeddings)
        print(f"PyTorch graph saved to {pt_file}")
    except ImportError:
        print("Warning: PyTorch not installed. Skipping PyTorch graph export.")
    
    
    visualization_file = os.path.join(config.OUTPUT_DIR, "graph_visualization.png")
    graph_builder.visualize_graph(output_file=visualization_file)
    

if __name__ == "__main__":

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    main()