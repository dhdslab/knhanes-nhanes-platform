# -*- coding: utf-8 -*-
"""
evidence_rag.py — retrieval-augmented evidence over the generated corpus
========================================================================

The platform produces a large corpus of self-contained reports (the merged
``Supplementary_Data_S*.docx`` bundles, individual ``.docx`` reports, the manuscript,
and the machine-readable association manifests). This module turns that corpus into a
searchable evidence base and lets a **Llama-3.3 agent read those documents and emit
evidence with citations**.

Flow
----
    corpus (.docx + manifest .csv)
        -> parse into evidence chunks (per report / per association row)
        -> retrieve the passages relevant to a question (BM25, or Ollama embeddings)
        -> Llama-3.3 synthesises an answer grounded ONLY in the retrieved passages,
           quoting numbers verbatim and citing each one back to its source report

Invariant
---------
The evidence agent may only **quote** numbers that already appear in the retrieved
passages (which were computed by the deterministic core). It never computes or alters a
number, and it must say the corpus has no direct evidence when retrieval finds none.

Runs anywhere
-------------
Retrieval defaults to a pure-Python BM25 index (no GPU, no extra dependencies), so the
whole thing is testable offline. If Ollama is available it is used for answer synthesis
(and, with ``--embeddings``, for dense retrieval); otherwise the agent returns an
extractive answer built from the top passages. Every result records its ``source``
("llm" or "fallback").

CLI
---
    python evidence_rag.py --demo
    python evidence_rag.py --corpus ../suppl --ask "Is uric acid associated with CKD?"
    python evidence_rag.py --corpus ../manuscript --ask "..." --embeddings
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Tuple

from agents import LlamaAgent, OllamaClient, OllamaUnavailable

_TOKEN = re.compile(r"[a-z0-9]+")

# Structural/query words that carry no retrieval signal. Filtered from both the index and
# the query so that, e.g., "X and Y" in a report does not match the "and" in a question.
_STOP = {
    "a", "an", "and", "or", "the", "in", "of", "to", "for", "with", "is", "are", "was",
    "were", "be", "on", "by", "at", "as", "that", "this", "it", "its", "from", "between",
    "both", "associated", "association", "what", "which", "does", "do", "how", "there",
    "any", "vs", "versus", "study", "survey", "n",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


# ─────────────────────────────────────────────────────────────────────────────
# Evidence chunk
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    id: str
    text: str
    source: str                       # human-readable citation locator
    meta: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────
def _iter_block_items(doc):
    """Yield paragraphs and tables of a python-docx Document in document order."""
    from docx.document import Document as _Doc
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = doc.element.body if isinstance(doc, _Doc) else doc
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _table_text(table) -> str:
    rows = []
    for row in table.rows:
        rows.append(" | ".join(c.text.strip() for c in row.cells))
    return "; ".join(r for r in rows if r.strip())


def parse_docx(path: str, max_chars: int = 1600) -> List[Chunk]:
    """Chunk a .docx into evidence passages, preserving the nearest heading as context.

    Works on individual reports, the merged supplements, and the manuscript. A new chunk
    starts at each heading and whenever a chunk would exceed ``max_chars``.
    """
    from docx import Document

    doc = Document(path)
    base = os.path.basename(path)
    chunks: List[Chunk] = []
    heading = ""
    buf: List[str] = []
    n = 0

    def flush():
        nonlocal buf, n
        text = "\n".join(x for x in buf if x.strip()).strip()
        buf = []
        if not text:
            return
        survey = ""
        m = re.search(r"\b(KNHANES|NHANES)\b", text)
        if m:
            survey = m.group(1)
        loc = f"{base}" + (f" > {heading}" if heading else "")
        chunks.append(Chunk(id=f"{base}#{n}", text=(f"{heading}\n{text}" if heading else text),
                            source=loc, meta={"file": base, "heading": heading, "survey": survey,
                                              "kind": "docx"}))
        n += 1

    for block in _iter_block_items(doc):
        cls = block.__class__.__name__
        if cls == "Paragraph":
            style = (block.style.name or "") if block.style else ""
            is_heading = ("Heading" in style) or style in ("Title", "SDataOutcome")
            txt = block.text.strip()
            if not txt:
                continue
            if is_heading:
                flush()
                heading = txt
                continue
            buf.append(txt)
        else:  # Table
            buf.append(_table_text(block))
        if sum(len(x) for x in buf) > max_chars:
            flush()
    flush()
    return chunks


def _get(row: Dict[str, str], *keys, default="") -> str:
    for k in keys:
        if k in row and row[k] != "":
            return row[k]
    return default


_LAB_CACHE: Optional[Dict[str, object]] = None


def _label(code: str) -> str:
    """Expand a variable code (e.g. ``ckd``) to its readable label so that
    natural-language questions retrieve it. Uses the platform's variable registry when
    importable, otherwise returns the code unchanged."""
    global _LAB_CACHE
    if _LAB_CACHE is None:
        try:
            import factory_core as fc  # heavy import; only attempted on real manifests
            _LAB_CACHE = dict(fc.VARS)
        except Exception:
            _LAB_CACHE = {}
    v = _LAB_CACHE.get(code)
    return v[0] if isinstance(v, (tuple, list)) and v else code


def parse_manifest_csv(path: str, survey: Optional[str] = None) -> List[Chunk]:
    """Turn an association manifest / results CSV into one precise evidence chunk per row.

    Accepts both the short manifest header (exp,out,measure,est,lo,hi,p,n,adj,q) and the
    consolidated header (survey,exposure,outcome,measure,estimate,ci_low,ci_high,...).
    """
    base = os.path.basename(path)
    if survey is None:
        m = re.search(r"(KNHANES|NHANES)", base)
        survey = m.group(1) if m else ""
    chunks: List[Chunk] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            exp = _get(row, "exposure", "exp")
            out = _get(row, "outcome", "out")
            measure = _get(row, "measure") or "estimate"
            est = _get(row, "estimate", "est")
            lo = _get(row, "ci_low", "lo")
            hi = _get(row, "ci_high", "hi")
            q = _get(row, "fdr_q", "q")
            n = _get(row, "n_analytic", "n")
            adj = _get(row, "adjustment", "adj")
            sv = _get(row, "survey") or survey
            try:
                est_s = f"{float(est):.2f}"
                ci_s = f"{float(lo):.2f} to {float(hi):.2f}"
            except ValueError:
                est_s, ci_s = est, f"{lo} to {hi}"
            unit = "odds ratio" if measure.upper() == "OR" else ("beta" if measure else "estimate")
            exp_l, out_l = _label(exp), _label(out)
            text = (f"In {sv}, {exp_l} and {out_l}: {unit} {est_s} (95% CI {ci_s}), "
                    f"FDR q {q}, n {n}." + (f" Adjusted for {adj}." if adj and adj != 'unadjusted' else ""))
            chunks.append(Chunk(id=f"{base}#{i}", text=text, source=f"{base} row {i + 1}",
                                meta={"file": base, "survey": sv, "exposure": exp, "outcome": out,
                                      "measure": measure, "estimate": est, "ci_low": lo, "ci_high": hi,
                                      "fdr_q": q, "n": n, "kind": "manifest"}))
    return chunks


def build_corpus(paths: Iterable[str], max_docx: Optional[int] = None) -> List[Chunk]:
    """Build a chunk list from files and/or directories.

    Directories are scanned for ``*.docx`` and ``*.csv`` (manifests / results). The huge
    merged association bundles are supported but slow to parse; pointing at the manifests
    is the fastest, most precise source.
    """
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.docx"), recursive=True))
            files += sorted(glob.glob(os.path.join(p, "**", "*manifest*.csv"), recursive=True))
            files += sorted(glob.glob(os.path.join(p, "**", "results*.csv"), recursive=True))
        else:
            files.append(p)
    chunks: List[Chunk] = []
    n_docx = 0
    for f in files:
        low = f.lower()
        try:
            if low.endswith(".docx") and not os.path.basename(f).startswith("~$"):
                if max_docx is not None and n_docx >= max_docx:
                    continue
                chunks += parse_docx(f)
                n_docx += 1
            elif low.endswith(".csv"):
                chunks += parse_manifest_csv(f)
        except Exception as exc:  # noqa: BLE001 - keep going on a bad file
            print(f"[evidence] skipped {f}: {exc}")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────
class BM25:
    """Pure-Python BM25 lexical retriever (no dependencies, deterministic, testable)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: List[Chunk] = []
        self.docs: List[List[str]] = []
        self.df: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0

    def fit(self, chunks: List[Chunk]) -> "BM25":
        self.chunks = chunks
        self.docs = [tokenize(c.text) for c in chunks]
        self.df = Counter()
        for d in self.docs:
            for term in set(d):
                self.df[term] += 1
        N = max(1, len(self.docs))
        self.idf = {t: math.log(1 + (N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}
        self.avgdl = (sum(len(d) for d in self.docs) / N) if self.docs else 0.0
        return self

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        q = tokenize(query)
        scores: List[Tuple[int, float]] = []
        for i, d in enumerate(self.docs):
            if not d:
                continue
            tf = Counter(d)
            dl = len(d)
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                f = tf[term]
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.chunks[i], sc) for i, sc in scores[:k]]


class EmbeddingIndex:
    """Optional dense retriever over Ollama embeddings; falls back to BM25 on failure."""

    def __init__(self, client: OllamaClient, embed_model: str = "nomic-embed-text"):
        self.client = client
        self.embed_model = embed_model
        self.chunks: List[Chunk] = []
        self.vecs: List[List[float]] = []
        self._bm25 = BM25()
        self.ok = False

    def fit(self, chunks: List[Chunk]) -> "EmbeddingIndex":
        self.chunks = chunks
        self._bm25.fit(chunks)
        try:
            self.vecs = [self.client.embed(c.text, model=self.embed_model) for c in chunks]
            self.ok = True
        except OllamaUnavailable:
            self.ok = False
        return self

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        num = sum(x * y for x, y in zip(a, b))
        da = math.sqrt(sum(x * x for x in a)) or 1.0
        db = math.sqrt(sum(y * y for y in b)) or 1.0
        return num / (da * db)

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.ok:
            return self._bm25.search(query, k)
        try:
            qv = self.client.embed(query, model=self.embed_model)
        except OllamaUnavailable:
            return self._bm25.search(query, k)
        scored = [(c, self._cos(qv, v)) for c, v in zip(self.chunks, self.vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Evidence agent
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Citation:
    source: str
    snippet: str
    score: float
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class EvidenceAnswer:
    question: str
    answer: str
    citations: List[Citation]
    source: str            # "llm" | "fallback"

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        return d

    def render(self) -> str:
        lines = [f"Q: {self.question}", "", self.answer, "", f"Evidence ({self.source}):"]
        for i, c in enumerate(self.citations, 1):
            lines.append(f"  [{i}] {c.source}")
            lines.append(f"      {c.snippet}")
        return "\n".join(lines)


class EvidenceRAGAgent(LlamaAgent):
    """Reads the retrieved report passages and emits an evidence-grounded answer.

    With Llama-3.3 available it synthesises a cited answer; otherwise it returns an
    extractive answer built from the top passages. In both cases every number shown comes
    verbatim from the corpus."""

    role = "evidence-rag"
    system = (
        "You are an evidence-synthesis assistant working over an audited re-analysis "
        "corpus of NHANES and KNHANES. Answer the question using ONLY the numbered "
        "evidence passages provided. Quote every number exactly as it appears; never "
        "invent, round, or recompute a number. Attribute each statement to its passage "
        "with a bracketed index like [1]. If the passages do not answer the question, "
        "reply that the corpus contains no direct evidence. Be concise."
    )

    def answer(self, question: str, retriever, k: int = 6) -> EvidenceAnswer:
        hits = retriever.search(question, k)
        citations = [Citation(source=c.source, snippet=c.text.replace("\n", " ")[:240],
                              score=float(s), meta=c.meta) for c, s in hits]
        if not hits:
            return EvidenceAnswer(question, "The corpus contains no direct evidence for this question.",
                                  [], "fallback")
        if self._use_llm():
            try:
                passages = "\n".join(f"[{i}] ({c.source}) {c.text}" for i, (c, _) in enumerate(hits, 1))
                ans = self.client.chat(self.system, f"Question: {question}\n\nEvidence passages:\n{passages}")
                return EvidenceAnswer(question, ans.strip(), citations, "llm")
            except Exception:
                pass
        # Extractive fallback: summarise how many passages matched and list them.
        lead = (f"The corpus contains {len(hits)} passage(s) relevant to this question. "
                "The most relevant evidence, quoted verbatim from the generated reports, is listed below.")
        body = "\n".join(f"[{i}] {c.text.splitlines()[-1][:240]}" for i, (c, _) in enumerate(hits, 1))
        return EvidenceAnswer(question, lead + "\n" + body, citations, "fallback")


# ─────────────────────────────────────────────────────────────────────────────
# Evidence base (corpus + retriever + agent)
# ─────────────────────────────────────────────────────────────────────────────
class EvidenceBase:
    def __init__(self, chunks: List[Chunk], client: Optional[OllamaClient] = None,
                 use_embeddings: bool = False, use_llm: bool = True):
        self.chunks = chunks
        self.client = client or OllamaClient()
        online = use_llm and self.client.available()
        if use_embeddings:
            self.retriever = EmbeddingIndex(self.client).fit(chunks)
        else:
            self.retriever = BM25().fit(chunks)
        self.agent = EvidenceRAGAgent(self.client, enabled=online)

    @classmethod
    def from_paths(cls, paths: Iterable[str], **kw) -> "EvidenceBase":
        return cls(build_corpus(paths), **kw)

    def ask(self, question: str, k: int = 6) -> EvidenceAnswer:
        return self.agent.answer(question, self.retriever, k)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([c.as_dict() for c in self.chunks], fh, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, **kw) -> "EvidenceBase":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls([Chunk(**c) for c in data], **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Demo + CLI
# ─────────────────────────────────────────────────────────────────────────────
def _demo_chunks() -> List[Chunk]:
    rows = [
        ("KNHANES", "Uric acid", "chronic kidney disease", "OR", "1.42", "1.34 to 1.51", "3e-20"),
        ("NHANES", "Uric acid", "chronic kidney disease", "OR", "1.43", "1.35 to 1.52", "5e-22"),
        ("NHANES", "Gamma-glutamyl transferase", "diabetes", "OR", "1.17", "1.11 to 1.23", "3e-8"),
        ("KNHANES", "Alanine aminotransferase", "metabolic syndrome", "OR", "1.88", "1.77 to 2.01", "2e-79"),
        ("KNHANES", "HOMA-IR", "hepatic steatosis", "OR", "3.83", "3.55 to 4.14", "5e-179"),
        ("NHANES", "Body mass index", "metabolic syndrome", "OR", "3.10", "2.80 to 3.43", "1e-40"),
    ]
    out = []
    for i, (sv, e, o, m, est, ci, q) in enumerate(rows):
        text = f"In {sv}, {e} and {o}: odds ratio {est} (95% CI {ci}), FDR q {q}."
        out.append(Chunk(id=f"demo#{i}", text=text, source=f"Supplementary_Data (demo) {sv} row {i + 1}",
                         meta={"survey": sv, "exposure": e, "outcome": o, "measure": m,
                               "estimate": est, "kind": "manifest"}))
    return out


def run_demo(args) -> int:
    base = EvidenceBase(_demo_chunks(), use_llm=not args.no_llm)
    where = "Llama-3.3" if base.agent.enabled else "extractive fallback"
    print(f"[evidence] retriever=BM25 · synthesis={where} · {len(base.chunks)} passages")
    questions = args.ask or [
        "Is uric acid associated with chronic kidney disease in both surveys?",
        "What predicts metabolic syndrome?",
    ]
    for qn in questions:
        ans = base.ask(qn, k=args.k)
        print("\n" + "=" * 78)
        print(ans.render())
    return 0


def run_corpus(args) -> int:
    if args.index and os.path.isfile(args.index):
        base = EvidenceBase.load(args.index, use_embeddings=args.embeddings, use_llm=not args.no_llm)
        print(f"[evidence] loaded index {args.index} ({len(base.chunks)} passages)")
    else:
        chunks = build_corpus(args.corpus, max_docx=args.max_docx)
        base = EvidenceBase(chunks, use_embeddings=args.embeddings, use_llm=not args.no_llm)
        print(f"[evidence] built corpus: {len(chunks)} passages from {args.corpus}")
        if args.index:
            base.save(args.index)
            print(f"[evidence] saved index to {args.index}")
    for qn in (args.ask or []):
        ans = base.ask(qn, k=args.k)
        print("\n" + "=" * 78)
        print(ans.render())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG evidence over the generated report corpus.")
    p.add_argument("--demo", action="store_true", help="run on a small synthetic corpus")
    p.add_argument("--corpus", nargs="*", help="files/dirs to index (e.g. ../suppl ../manuscript)")
    p.add_argument("--ask", nargs="*", help="question(s) to answer")
    p.add_argument("--index", help="path to save/load the parsed index (JSON)")
    p.add_argument("--k", type=int, default=6, help="passages to retrieve (default 6)")
    p.add_argument("--max-docx", type=int, default=None, help="cap number of .docx parsed")
    p.add_argument("--embeddings", action="store_true", help="use Ollama embeddings for retrieval")
    p.add_argument("--no-llm", action="store_true", help="force extractive (no synthesis)")
    p.add_argument("--model", default="llama3.3:70b")
    p.add_argument("--url", default="http://localhost:11434")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.demo:
        return run_demo(args)
    if not args.corpus:
        print("error: provide --corpus <dir/files> (or use --demo).")
        return 2
    return run_corpus(args)


if __name__ == "__main__":
    raise SystemExit(main())
