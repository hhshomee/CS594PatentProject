import os


DATA_DIR = "data"
OUTPUT_DIR = "output"
MODELS_DIR = "models"


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


CSV_FILE = os.path.join(DATA_DIR, "data_A61.csv")
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
GRAPH_FILE = os.path.join(OUTPUT_DIR, "graph.graphml")
VISUALIZATION_FILE = os.path.join(OUTPUT_DIR, "graph.png")

PYG_FILE = os.path.join(OUTPUT_DIR, "patent_graph.pt")
DGL_FILE = os.path.join(OUTPUT_DIR, "patent_graph.bin")


MODEL_NAME = "all-MiniLM-L6-v2"  

SIMILARITY_THRESHOLD = 0.7
METRIC = "cosine"
COMPREHENSIVE_EDGE_CSV = os.path.join(OUTPUT_DIR, "patent_edges_full.csv")