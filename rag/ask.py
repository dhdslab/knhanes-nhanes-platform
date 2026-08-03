# -*- coding: utf-8 -*-
"""Ask a plain-language question against the evidence reports.

Examples:
    python rag/ask.py "Is uric acid associated with hypertension in NHANES?"
    python rag/ask.py "odds ratio for BMI and diabetes" --survey KNHANES
    python rag/ask.py "which model predicts metabolic syndrome best?" --type ml
"""
import argparse
import rag_core as rc

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Search the evidence reports.")
    ap.add_argument("question", nargs="+", help="your question in plain language")
    ap.add_argument("--index", default=rc.DEFAULT_INDEX)
    ap.add_argument("-k", type=int, default=6, help="number of reports to retrieve")
    ap.add_argument("--survey", choices=["KNHANES", "NHANES"], help="restrict to one survey")
    ap.add_argument("--type", dest="rtype", choices=["association", "trend", "ml"],
                    help="restrict to a report type")
    ap.add_argument("--no-llm", action="store_true", help="force extractive answer (no Ollama)")
    ap.add_argument("--model", default="llama3.2", help="Ollama chat model to write the answer")
    a = ap.parse_args()
    q = " ".join(a.question)
    ix = rc.Index(a.index)
    res = rc.answer(q, index=ix, k=a.k, use_llm=(False if a.no_llm else "auto"),
                    llm_model=a.model, survey=a.survey, rtype=a.rtype)
    print("\n" + res["answer"] + "\n")
    print("-" * 60)
    print("Sources:")
    for s in res["sources"]:
        print(f"  [{s['score']:.3f}] {s['path']}")
