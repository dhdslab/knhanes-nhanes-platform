# Evidence RAG over the generated corpus

`evidence_rag.py` turns the platform's own output into a searchable evidence base: the
association manifests, the individual `.docx` reports and, if given, the merged
Supplementary Data bundles. An agent reads those documents and answers questions with cited
evidence.

```
corpus (.docx + manifest .csv)
   -> parse into evidence chunks (per report / per association row)
   -> retrieve the passages relevant to a question  (BM25, or Ollama embeddings)
   -> Llama 3.3 writes an answer grounded only in the retrieved passages,
      quoting every number verbatim and citing its source report
```

For a simpler, non-specialist interface over the same corpus (web page or command line,
TF-IDF or `bge-m3` retrieval), see [`rag/`](../rag/README.md).

## The rule

The evidence agent may only **quote** numbers that already appear in the retrieved passages,
which the deterministic core computed. It never computes or alters a number. When retrieval
finds nothing relevant, it says the corpus has no direct evidence.

## Components

| Piece | Role |
|---|---|
| `parse_docx`, `parse_manifest_csv` | Parse the corpus into `Chunk`s with source locators |
| `BM25` | Pure-Python lexical retriever (default; no GPU, deterministic, testable) |
| `EmbeddingIndex` | Optional dense retriever over Ollama embeddings (falls back to BM25) |
| `EvidenceRAGAgent` | Writes a cited answer with Llama 3.3, or an extractive answer offline |
| `EvidenceBase` | Ties corpus, retriever and agent together, with `save`/`load` of the parsed index |

Variable codes in the manifests (`ckd`, `dm`) are expanded to their readable labels so that
natural-language questions retrieve them.

## Use it

```bash
# Synthetic corpus (no data, no GPU):
python evidence_rag.py --demo

# The released association manifests (fast and precise; 7,478 evidence rows):
python evidence_rag.py --corpus suppl --max-docx 0 \
    --ask "Is serum uric acid associated with chronic kidney disease?"

# Read the Word reports too, with dense retrieval and synthesis by Llama 3.3:
python evidence_rag.py --corpus suppl --embeddings \
    --ask "What are the strongest predictors of metabolic syndrome in both surveys?"

# Persist the parsed index and reuse it:
python evidence_rag.py --corpus suppl --index evidence_index.json --ask "..."
```

Each passage in the answer carries its locator, for example
`_manifest_association_NHANES.csv row 975` for the NHANES serum uric acid and kidney
phenotype row. With Ollama running, the same call returns a written, cited paragraph instead
of the bare passages, and every number in it is quoted from the corpus.

## Notes

- Parsing all 7,584 Word reports is slow. Point `--corpus` at the manifests (`--max-docx 0`)
  for speed and precision, or cap the number of reports parsed with `--max-docx`.
- Retrieval defaults to BM25 so that the tool runs and is tested everywhere. `--embeddings`
  improves semantic recall when an embedding model such as `nomic-embed-text` is pulled.
