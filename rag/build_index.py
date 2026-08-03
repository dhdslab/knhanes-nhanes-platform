# -*- coding: utf-8 -*-
"""Build the search index over the evidence reports.

Usage:
    python rag/build_index.py                       # default: suppl/ , TF-IDF backend
    python rag/build_index.py --backend ollama      # semantic embeddings via Ollama (bge-m3)
    python rag/build_index.py --suppl path/to/suppl --index path/to/_index
"""
import argparse
import rag_core as rc

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Index the KNHANES/NHANES evidence reports.")
    ap.add_argument("--suppl", default=rc.DEFAULT_SUPPL, help="folder with the report .docx files")
    ap.add_argument("--index", default=rc.DEFAULT_INDEX, help="output folder for the index")
    ap.add_argument("--backend", choices=["tfidf", "ollama"], default="tfidf",
                    help="tfidf = no setup (default); ollama = bge-m3 semantic embeddings")
    ap.add_argument("--embed-model", default="bge-m3")
    a = ap.parse_args()
    n = rc.build_index(a.suppl, a.index, backend=a.backend, embed_model=a.embed_model)
    print(f"Done. Indexed {n} reports -> {a.index}")
