"""
build_index.py – Run once to build ChromaDB vector index from extracted graphs.

Usage:
    python build_index.py [--graph_dir ./data/graphs_new] [--chroma_dir ./chroma_db]
"""

import argparse
from src.retriever import RuRobertaEmbedder, VectorIndex, GraphIndex
from sentence_transformers import SentenceTransformer
parser = argparse.ArgumentParser()
parser.add_argument("--graph_dir",  default="./data/graphs_new")
parser.add_argument("--chroma_dir", default="./chroma_db")
parser.add_argument("--embed_model", default="ai-forever/FRIDA")
parser.add_argument("--device",     default="cpu")
args = parser.parse_args()

# embedder = SentenceTransformer(args.embed_model, args.device)
# idx      = VectorIndex(args.chroma_dir, embedder)
# idx.index_graphs(args.graph_dir)

graphid = GraphIndex(args.graph_dir)

print("Done.")
