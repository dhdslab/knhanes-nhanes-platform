# -*- coding: utf-8 -*-
"""Harvest the abstract-level facts needed to compare the corpus with the
published literature qualitatively: direction of association, reported effect
size, and reported discrimination for prediction models.

Two harvests, both cached so the script can be interrupted and resumed.

  A. association papers   the 462 papers whose title states an exposure and an
                          outcome that both fall inside the platform vocabulary
  B. prediction papers    for each of the 27 harmonized clinical outcomes, the
                          NHANES/KNHANES papers that report a discrimination
                          statistic (AUC, AUROC, C-statistic)

Nothing here is mapped to a manuscript number; the products are direction signs,
effect sizes and AUC values, which the figure script compares with the corpus.
"""
import os, sys, re, json, time, urllib.request, urllib.parse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "knhanes_platform"))
import pubmed_terms as PT
import factory_core as fc

E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
F = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ABS_A = os.path.join(HERE, "abstracts_association.csv")
ABS_B = os.path.join(HERE, "abstracts_prediction.csv")
SLEEP = 0.40

PRED_TERMS = ('("area under the curve"[tiab] OR AUC[tiab] OR AUROC[tiab] OR '
              '"area under the receiver"[tiab] OR "C-statistic"[tiab] OR '
              '"c statistic"[tiab] OR "concordance index"[tiab] OR '
              '"machine learning"[tiab] OR "prediction model"[tiab] OR nomogram[tiab])')


def _get(url, tries=4, timeout=120):
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8", "replace")
        except Exception:
            time.sleep(1.5 * (k + 1))
    return ""


def fetch_abstracts(pmids, label):
    rows, step = [], 200
    for i in range(0, len(pmids), step):
        xml = _get(F + "?db=pubmed&retmode=xml&rettype=abstract&tool=knhanes-nhanes-platform&id="
                   + ",".join(pmids[i:i + step]))
        for art in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.S):
            pm = re.search(r"<PMID[^>]*>(\d+)</PMID>", art)
            ti = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", art, re.S)
            ab = " ".join(re.sub("<[^>]+>", " ", a)
                          for a in re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", art, re.S))
            yr = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", art, re.S)
            if pm:
                rows.append((pm.group(1), yr.group(1) if yr else "",
                             re.sub("<[^>]+>", "", ti.group(1)).strip() if ti else "",
                             re.sub(r"\s+", " ", ab).strip()))
        print(f"    {label}: {min(i+step, len(pmids))}/{len(pmids)}", flush=True)
        time.sleep(SLEEP)
    return pd.DataFrame(rows, columns=["pmid", "year", "title", "abstract"])


def harvest_a():
    if os.path.exists(ABS_A):
        d = pd.read_csv(ABS_A); print(f"  reusing {len(d)} association abstracts"); return d
    X = pd.read_csv(os.path.join(HERE, "literature_studied.csv"))
    B = X[X.var_a.notna() & X.var_b.notna() & (X.var_a != X.var_b)]
    pmids = sorted(B.pmid.astype(str).unique())
    print(f"  fetching abstracts for {len(pmids)} association papers")
    d = fetch_abstracts(pmids, "assoc")
    d.to_csv(ABS_A, index=False, encoding="utf-8-sig")
    return d


def harvest_b():
    if os.path.exists(ABS_B):
        d = pd.read_csv(ABS_B); print(f"  reusing {len(d)} prediction abstracts"); return d
    outs = [o for o in fc.OUTCOMES if fc.typ(o) == "b" and PT.concept(o)]
    print(f"  prediction harvest over {len(outs)} clinical outcomes")
    frames = []
    for o in outs:
        q = f"{PT.SURVEY} AND {PT.concept(o)} AND {PRED_TERMS}"
        js = _get(E + "?db=pubmed&retmode=json&retmax=400&tool=knhanes-nhanes-platform&term="
                  + urllib.parse.quote(q), timeout=90)
        try:
            ids = json.loads(js)["esearchresult"].get("idlist", [])
        except Exception:
            ids = []
        time.sleep(SLEEP)
        if not ids:
            print(f"    {o}: 0"); continue
        d = fetch_abstracts(ids, o)
        d["outcome"] = o
        frames.append(d)
        print(f"    {o}: {len(ids)} papers, {len(d)} abstracts", flush=True)
    D = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    D.to_csv(ABS_B, index=False, encoding="utf-8-sig")
    return D


if __name__ == "__main__":
    print("A. association abstracts")
    a = harvest_a(); print(f"   {len(a)} rows, {a.abstract.str.len().gt(50).sum()} with a real abstract")
    print("B. prediction abstracts")
    b = harvest_b()
    if len(b):
        print(f"   {len(b)} rows over {b.outcome.nunique()} outcomes, "
              f"{b.pmid.nunique()} unique papers")
