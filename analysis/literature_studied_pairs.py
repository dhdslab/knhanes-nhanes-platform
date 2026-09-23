# -*- coding: utf-8 -*-
"""What has the NHANES/KNHANES literature actually STUDIED, and how much of it
does the corpus contain?

The first attempt at this counted papers that merely co-mention two concepts.
That measures topic co-occurrence, not studied associations: the top hit for
"obesity x overweight" was a paper on obesity and periodontitis. This script
instead parses the exposure-outcome pair out of the title, which works because
this literature is formulaic - the very point the manuscript makes. Roughly 40%
of titles carry an explicit "association between A and B" construction.

Outputs
  literature_titles.csv        pmid, year, title, extracted A, extracted B
  literature_studied.csv       one row per studied pair, mapped to our vocabulary
and a report separating two different questions:
  (i)  within our declared variable vocabulary, how exhaustive is the corpus?
  (ii) how much of the literature's pair space lies outside that vocabulary?
"""
import os, sys, re, json, time, urllib.request, urllib.parse, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repository root
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import pubmed_terms as PT
import factory_core as fc

E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
F = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TITLES = os.path.join(HERE, "literature_titles.csv")
STUDIED = os.path.join(HERE, "literature_studied.csv")

# "association of A and B with C" lists two CO-EXPOSURES and one outcome. Treating
# A and B as an exposure-outcome pair is wrong, and wrong in a biased way: co-exposures
# are usually related constructs, so the false pairs pile up exactly on the pairs the
# leakage guard blocks. Such titles are split into (A,C) and (B,C) instead.
_END = r'(?:[:;.,]|\bin\b|\bamong\b|\bwith\b|$)'
_END2 = r'(?:[:;.,]|\bin\b|\bamong\b|$)'

MULTI = [re.compile(p, re.I) for p in [
    r'\bassociations? (?:between|of) (.{3,60}?) and (.{3,60}?) with (.{3,70}?)' + _END2,
    r'\brelationships? (?:between|of) (.{3,60}?) and (.{3,60}?) with (.{3,70}?)' + _END2,
    r'^(.{3,60}?) and (.{3,60}?) (?:are|were|is|was) associated with (.{3,70}?)' + _END2,
]]

PATTERNS = [re.compile(p, re.I) for p in [
    r'\bassociations? (?:between|of) (.{3,70}?) (?:and|with) (.{3,70}?)' + _END,
    r'\brelationships? (?:between|of) (.{3,70}?) (?:and|with) (.{3,70}?)' + _END,
    r'\b(?:impact|effect|influence) of (.{3,70}?) on (.{3,70}?)' + _END2,
    r'^(.{3,70}?) (?:is|are|was|were) (?:positively |inversely |negatively )?associated with (.{3,70}?)' + _END2,
    r'\b(.{3,60}?) and (?:its |the )?(?:risk of |prevalence of )(.{3,60}?)' + _END2,
]]

# a phrase that is itself a list ("A, B, and C") cannot be resolved to one variable
LISTY = re.compile(r',\s|\band\b', re.I)


# A phrase that runs across a population qualifier is not a variable: in
# "... with steatosis in patients with metabolic syndrome" the trailing clause
# names who was studied, not what was measured.
SPANS = re.compile(r'\b(?:in|among|with)\b.*\b(?:patients|adults|individuals|'
                   r'participants|subjects|women|men|children|adolescents)\b', re.I)
# Composite indices are their own construct, not the component they contain.
COMPOSITE = re.compile(r'\b(?:ratio|index|score|indices)\b', re.I)


def usable_phrase(phrase, var):
    """False when the phrase cannot be trusted to denote `var`."""
    if var is None:
        return True                     # unmapped phrases are counted as outside
    if SPANS.search(phrase):
        return False
    if COMPOSITE.search(phrase):
        keys = KEYS.get(var, [])
        p = re.sub(r"[^a-z0-9 ]", " ", phrase.lower()).strip()
        p = re.sub(r"\s+", " ", p)
        if not any(p == k for k in keys):
            return False
    return True


def extract_pairs(title):
    """Return the exposure-outcome pairs a title states, as (a_phrase, b_phrase)."""
    for p in MULTI:
        m = p.search(title)
        if m:
            a, b, c = (g.strip() for g in m.groups())
            if any(re.search(r'\b(?:in|among)\b', g) for g in (a, b)):
                break          # the match ran across a population qualifier
            return [(a, c), (b, c)]
    for p in PATTERNS:
        m = p.search(title)
        if m:
            a, b = m.group(1).strip(), m.group(2).strip()
            if LISTY.search(a) or LISTY.search(b):
                return []              # unresolvable list construction
            return [(a, b)]
    return []


# surface forms used to map a free-text phrase onto one of our 70 variables
KEYS = {v: sorted({s.strip().strip('"').lower()
                   for s in re.findall(r'"([^"]+)"\[tiab\]|(\w[\w-]{2,})\[tiab\]', t)
                   for s in ([s[0] or s[1]])}, key=len, reverse=True)
        for v, t in PT.TERMS.items()}
EXTRA = {
 # Disease-polarity variables keep only phrases that name the abnormality.
 "dm": ["type 2 diabetes", "t2dm", "diabetes mellitus"],
 "mets": ["metabolic syndrome", "mets"],
 "steatosis": ["nafld", "mafld", "fatty liver", "hepatic steatosis"],
 "obesity": ["obesity", "obese"],
 "htn": ["hypertension"],
 "ckd": ["chronic kidney disease"],
 "dyslipidemia": ["dyslipidemia", "hyperlipidemia"],
 "anemia": ["anemia", "anaemia"],
 "sarcopenia": ["sarcopenia", "low muscle mass", "low skeletal muscle mass",
                "low lean mass", "low appendicular"],
 "osteoporosis": ["osteoporosis", "osteoporotic", "low bone mineral density",
                  "low bone mass", "reduced bone mineral density"],
 "vitd_deficiency": ["vitamin d deficiency", "vitamin d insufficiency",
                     "hypovitaminosis d", "low vitamin d"],
 "low_hdl": ["low hdl", "low high-density lipoprotein"],
 "insulin_resistance": ["insulin resistance"],
 "homa_ir": ["homa-ir"],
 # Continuous measurements keep the neutral phrase, so a "low X" title maps here
 # and its direction is flipped by the polarity rule in extract_and_compare.py.
 "bmi": ["body mass index", "bmi"],
 "wc": ["waist circumference"],
 "vitd": ["vitamin d", "25-hydroxyvitamin d", "serum vitamin d"],
 "hdl": ["hdl cholesterol", "high-density lipoprotein cholesterol"],
 "asmi": ["skeletal muscle mass index", "muscle mass index", "appendicular lean mass index"],
 "asmm_kg": ["skeletal muscle mass", "muscle mass", "appendicular lean mass",
             "appendicular skeletal muscle"],
 "fn_bmd": ["femoral neck bone mineral density", "femoral neck bmd"],
 "ls_bmd": ["lumbar spine bone mineral density", "lumbar spine bmd"],
 "egfr": ["glomerular filtration rate", "egfr", "kidney function"],
 "uric": ["uric acid", "hyperuricemia"],
 "alcohol": ["alcohol"],
 "smoking": ["smoking", "smokers"],
}
# "bone mineral density" on its own is a continuous measurement; it must not fall
# through to osteoporosis. It is ambiguous between sites, so it maps to the
# femoral-neck variable, the one the surveys report for every cycle.
EXTRA["fn_bmd"].append("bone mineral density")

for k, v in EXTRA.items():
    KEYS[k] = sorted(set(KEYS.get(k, []) + v), key=len, reverse=True)


def match_var(phrase):
    p = " " + re.sub(r"[^a-z0-9\- ]", " ", phrase.lower()) + " "
    p = re.sub(r"\s+", " ", p)
    best = None
    for v, keys in KEYS.items():
        for k in keys:
            if f" {k} " in p or p.strip() == k:
                if best is None or len(k) > best[1]:
                    best = (v, len(k))
    return best[0] if best else None


def _pmids_for_year(year):
    """esearch capped per year; every year of this literature is well under 10,000."""
    term = f"{PT.SURVEY} AND (\"{year}\"[dp])"
    url = (E + "?db=pubmed&retmode=json&retmax=10000&tool=knhanes-nhanes-platform&term="
           + urllib.parse.quote(term))
    for attempt in range(4):
        try:
            r = json.load(urllib.request.urlopen(url, timeout=90))["esearchresult"]
            return r.get("idlist", []), int(r.get("count", 0))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return [], 0


def _titles_for(ids):
    rows, step = [], 300
    for i in range(0, len(ids), step):
        url = (F + "?db=pubmed&retmode=xml&rettype=abstract&tool=knhanes-nhanes-platform&id="
               + ",".join(ids[i:i + step]))
        for attempt in range(4):
            try:
                xml = urllib.request.urlopen(url, timeout=150).read().decode("utf-8", "replace")
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            continue
        for art in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.S):
            pm = re.search(r"<PMID[^>]*>(\d+)</PMID>", art)
            ti = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", art, re.S)
            yr = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", art, re.S)
            if pm and ti:
                rows.append((pm.group(1), yr.group(1) if yr else "",
                             re.sub("<[^>]+>", "", ti.group(1)).strip()))
        time.sleep(0.4)
    return rows


def fetch_titles():
    """NCBI refuses retstart beyond 10,000, so the query is partitioned by
    publication year. Every year is retrieved in full; nothing is sampled."""
    if os.path.exists(TITLES):
        d = pd.read_csv(TITLES)
        print(f"reusing {TITLES}: {len(d)} titles")
        return d
    all_rows, expected = [], 0
    for year in range(2003, 2027):
        ids, n = _pmids_for_year(year)
        expected += n
        time.sleep(0.4)
        if not ids:
            print(f"  {year}: 0"); continue
        rows = _titles_for(ids)
        all_rows += rows
        print(f"  {year}: {n} indexed, {len(rows)} titles retrieved "
              f"(running total {len(all_rows)})", flush=True)
    d = pd.DataFrame(all_rows, columns=["pmid", "year", "title"]).drop_duplicates("pmid")
    d.to_csv(TITLES, index=False, encoding="utf-8-sig")
    print(f"wrote {TITLES}: {len(d)} unique titles (year-sum {expected})")
    return d


def main():
    T = fetch_titles()
    ex = []
    for r in T.itertuples():
        t = str(r.title)
        for a, b in extract_pairs(t):
            va, vb = match_var(a), match_var(b)
            if not usable_phrase(a, va):
                va = None
            if not usable_phrase(b, vb):
                vb = None
            ex.append((r.pmid, r.year, t, a, b, va, vb))
    X = pd.DataFrame(ex, columns=["pmid", "year", "title", "phrase_a", "phrase_b", "var_a", "var_b"])
    X.to_csv(STUDIED, index=False, encoding="utf-8-sig")
    print(f"\ntitles with an extractable association pair: {len(X)}/{len(T)} "
          f"({100*len(X)/len(T):.1f}%)")

    both = X[X.var_a.notna() & X.var_b.notna() & (X.var_a != X.var_b)].copy()
    one = X[(X.var_a.notna() ^ X.var_b.notna())]
    none = X[X.var_a.isna() & X.var_b.isna()]
    print(f"  both sides inside our 70-variable vocabulary : {len(both):>6} "
          f"({100*len(both)/len(X):.1f}%)")
    print(f"  one side inside                              : {len(one):>6} "
          f"({100*len(one)/len(X):.1f}%)")
    print(f"  neither side inside                          : {len(none):>6} "
          f"({100*len(none)/len(X):.1f}%)")

    K = pd.read_csv(os.path.join(ROOT, "suppl", "_manifest_association_KNHANES.csv"))
    N = pd.read_csv(os.path.join(ROOT, "suppl", "_manifest_association_NHANES.csv"))
    inK = {frozenset((r.exp, r.out)) for r in K.itertuples()}
    inN = {frozenset((r.exp, r.out)) for r in N.itertuples()}
    corpus = inK | inN
    strength = {}
    for d in (K, N):
        o = d[(d.measure == "OR") & (d.est > 0)]
        for r in o.itertuples():
            k = frozenset((r.exp, r.out)); v = abs(np.log(r.est))
            strength[k] = max(strength.get(k, -1), v)

    both["pair"] = [frozenset((a, b)) for a, b in zip(both.var_a, both.var_b)]
    freq = collections.Counter(both.pair)
    studied = pd.DataFrame([{"pair": p, "papers": n,
                             "in_corpus": p in corpus,
                             "abs_log_or": strength.get(p)} for p, n in freq.items()])
    print("\n" + "=" * 72)
    print("(i) EXHAUSTIVENESS WITHIN OUR DECLARED VOCABULARY")
    print(f"   distinct pairs the literature has studied, both sides in our vocabulary : "
          f"{len(studied)}")
    print(f"   of those, present in the corpus                                        : "
          f"{int(studied.in_corpus.sum())}  ({100*studied.in_corpus.mean():.1f}%)")
    miss = studied[~studied.in_corpus].sort_values("papers", ascending=False)
    print(f"   studied pairs the corpus does NOT contain                              : {len(miss)}")
    for r in miss.head(10).itertuples():
        a, b = tuple(r.pair); print(f"       {r.papers:>4} papers  {fc.lab(a)[:32]:<33} x {fc.lab(b)[:32]}")
    print(f"\n   corpus pairs never studied in the literature                          : "
          f"{len(corpus) - int(studied.in_corpus.sum())}  of {len(corpus)} corpus pairs")

    print("\n(ii) HOW MUCH OF THE LITERATURE LIES OUTSIDE THAT VOCABULARY")
    print(f"   association papers whose pair is fully covered by our vocabulary : "
          f"{len(both)} of {len(X)}  ({100*len(both)/len(X):.1f}%)")
    print("   the rest study exposures or outcomes the platform does not define")

    print("\n(iii) DOES STUDY FREQUENCY TRACK MEASURED EFFECT STRENGTH?")
    s = studied[studied.abs_log_or.notna()]
    if len(s) > 20:
        from scipy.stats import spearmanr
        rho, p = spearmanr(s.papers, s.abs_log_or)
        print(f"   Spearman rho (papers vs |log OR|) : {rho:+.3f}  (p={p:.2e}, n={len(s)})")

    print("\n(iv) PAIRS THE LEAKAGE GUARD BLOCKS THAT THE LITERATURE ACTUALLY STUDIED")
    blocked = studied[~studied.in_corpus]
    print(f"   distinct blocked-but-studied pairs : {len(blocked)}")
    print(f"   papers on them                     : {int(blocked.papers.sum())}")
    print("=" * 72)
    studied["a"] = [fc.lab(tuple(p)[0]) for p in studied.pair]
    studied["b"] = [fc.lab(tuple(p)[1]) for p in studied.pair]
    studied.drop(columns=["pair"]).to_csv(
        os.path.join(HERE, "literature_studied_pairs.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
