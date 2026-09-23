# -*- coding: utf-8 -*-
"""Supplementary generator (v2) — three analysis types with their own combinatorics:

  suppl/<dataset>/association/<Outcome>/<Exposure>__<Outcome>_<dataset>.docx
        one file per exposure x outcome pair; each exposure adjusted for the OUTCOME's
        literature-based confounder set (minus the exposure's own family to avoid over-adjustment);
        definitional / same-family / over-influential variables are excluded.
  suppl/<dataset>/trend/<Outcome>__trend_<dataset>.docx
        one file per clinical outcome (prevalence over survey years; KNHANES has >=2 cycles).
  suppl/<dataset>/ml/<Outcome>__prediction_<dataset>.docx
        one file per clinical outcome (multivariable prediction with curated, leakage-free features).

Statistics are deterministic (R survey / scikit-learn). Prose is deterministic (thousands of files).
"""
import sys, os, re, time, traceback
PROJ=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ); os.chdir(PROJ)
import factory_core as fc
import trend_report, ml_report
import pandas as pd, numpy as np
from docx import Document

AGE=20
DEFS={**fc.DEFAULT_DEFS,"pop":{"age_min":AGE}}
# The study kept the corpus beside the code (../suppl); this repository ships it at ./suppl.
# Set SUPPL_DIR to regenerate somewhere else and compare with the released corpus.
SUPPL=os.path.abspath(os.environ.get("SUPPL_DIR",os.path.join(PROJ,"suppl")))
DATA={"KNHANES":(os.path.join(PROJ,"data","KNHANES"),
        ("07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24")),
      "NHANES":(os.path.join(PROJ,"data","NHANES"),("d","e","f","g","h","i","j","l"))}
ML_MODELS=["LogisticRegression","HistGradientBoosting","XGBoost"]   # fast, strong; bulk mode subsamples rows
ML_SAMPLE_N=6000
ASSOC_SAMPLE_N=50000   # for very large pooled multi-year data, screen associations on a random subsample
                       # (point estimates are stable; the app runs specific pairs on the full data)

# ── report-writer agent ───────────────────────────────────────────────────────
# When SUPPL_USE_LLM=1 the report-writer composes each report's prose from the
# frozen fact string produced by the deterministic core. fc.llm_prose enforces
# the numeric guard: any generated sentence containing a number that is not in
# the fact string is discarded and the deterministic template is written instead.
# The per-run tally in fc.LLM_PROSE_STATS is the audit record of that split.
USE_LLM = os.environ.get("SUPPL_USE_LLM","0")=="1"
LLM_MODEL = os.environ.get("SUPPL_LLM_MODEL","llama3.3:70b")
LLM_URL   = os.environ.get("SUPPL_LLM_URL","http://localhost:11434")
os.makedirs(SUPPL, exist_ok=True)
# append, so that importing this module (regen_ml_cvselect.py and regen_high_tg.py do)
# never truncates the log of an earlier run
LOG=open(os.path.join(SUPPL,"_run.log"),"a",encoding="utf-8")
def log(*a):
    m=" ".join(str(x) for x in a); print(m,flush=True); LOG.write(m+"\n"); LOG.flush()

def slug(v):
    s=fc.lab(v); s=re.sub(r"\(.*?\)","",s); s=s.split(",")[0]
    s=re.sub(r"[^0-9A-Za-z ]"," ",s); parts=[p for p in s.split() if p]
    return "".join(p[:1].upper()+p[1:] for p in parts) or re.sub(r"[^0-9A-Za-z]","",v) or "var"
def _src(ds):
    return {"KNHANES":"Korea National Health and Nutrition Examination Survey",
            "NHANES":"National Health and Nutrition Examination Survey"}.get(ds,ds)
def run_engine_retry(ana, ds, exps, outs, cov, tries=3, reuse=False):
    last=None
    for _ in range(tries):
        try: return fc.run_engine(ana, ds, exps, outs, cov, skip_table1=True, reuse_analytic=reuse)
        except Exception as e: last=e; time.sleep(0.5); reuse=False  # rewrite CSV on retry
    raise last
def wcell(ana, var, mask):
    x=pd.to_numeric(ana[var],errors="coerce"); w=pd.to_numeric(ana["wt_pool"],errors="coerce")
    m=mask & x.notna() & w.notna()
    if int(m.sum())==0: return "-"
    uniq=set(pd.Series(x[m]).dropna().unique())
    if fc.typ(var)=="b" or uniq<= {0,1,0.0,1.0}:
        return f"{100*np.average((x[m]==1).astype(float),weights=w[m]):.1f}%"
    return f"{np.average(x[m],weights=w[m]):.1f}"
def wsum(ana, var, ycol, binary_out):
    allm=pd.Series(True,index=ana.index)
    if binary_out:
        y=pd.to_numeric(ana[ycol],errors="coerce")
        return wcell(ana,var,allm), wcell(ana,var,y==0), wcell(ana,var,y==1)
    return wcell(ana,var,allm), None, None

# ─────────────────────────── ASSOCIATION ───────────────────────────
def gen_association(ds, d):
    if len(d)>ASSOC_SAMPLE_N:
        d=d.sample(ASSOC_SAMPLE_N,random_state=0).reset_index(drop=True)
        log(f"[{ds}] ASSOCIATION on random subsample of {ASSOC_SAMPLE_N} rows (large pooled data)")
    log(f"[{ds}] ASSOCIATION ...")
    outs=fc.outcome_candidates(ds)
    assoc=[]; meta={}; skipped=[]
    for oi,O in enumerate(outs):
        try:
            binary=fc.typ(O)=="b"
            cf=fc.confounders_for(ds,O)
            exps=fc.association_exposures(ds,O)
            if not exps: skipped.append((O,"no exposures")); continue
            # group exposures by adjustment set (over-adjustment avoided per family). Confounders are
            # collapsed into a couple of groups (age/sex crude or minimally adjusted) so the number of
            # engine calls stays small.
            cfset=set(cf); groups={}
            for e in exps:
                if e in cfset:
                    cov=() if e in ("age","men") else tuple(c for c in ("age","men") if c in cfset and c!=e)
                else:
                    cov=tuple(fc.adjustment_for(ds,O,e))
                groups.setdefault(cov,[]).append(e)
            # build the analytic ONCE (all exposures z-scored) and reuse the CSV across cov-groups
            first_ana=fc.build_analytic(d, exps, [O], [], AGE); n=len(first_ana)
            if binary:
                ev=int((pd.to_numeric(first_ana[O],errors="coerce")==1).sum())
                if ev<fc.MIN_EVENTS: skipped.append((O,f"low events {ev}")); continue
            rows=[]; wrote=False
            for adj,es in groups.items():
                _,res,_,_=run_engine_retry(first_ana, ds, es, [O], list(adj), reuse=wrote); wrote=True
                adjtxt=", ".join(fc.lab(c) for c in adj) if adj else "unadjusted"
                for _,r in res.iterrows():
                    rows.append(dict(exp=r.exposure[2:],out=O,measure=r.measure,est=r.est,lo=r.ci_low,hi=r.ci_high,
                                     p=r.p,n=n,adj=adjtxt))
            if not rows: skipped.append((O,"no fit")); continue
            # weighted prevalence + confounder descriptives (once per outcome, from first analytic)
            if binary:
                yb=(pd.to_numeric(first_ana[O],errors="coerce")==1).astype(float); w=pd.to_numeric(first_ana.wt_pool,errors="coerce")
                mm=yb.notna()&w.notna(); wprev=float(np.average(yb[mm],weights=w[mm]))*100
            else: wprev=float("nan")
            conf_desc={c:wsum(first_ana,c,O,binary) for c in cf if c in first_ana.columns}
            for r in rows:
                r["exp_desc"]=wsum(first_ana,r["exp"],O,binary) if r["exp"] in first_ana.columns else (None,None,None)
                assoc.append(r)
            meta[O]=dict(cf=cf,conf_desc=conf_desc,rat=fc.outcome_rationale(O),binary=binary,wprev=wprev)
            log(f"  [{ds} assoc {oi+1}/{len(outs)}] {O}: {len(rows)} exposures")
        except Exception as ex:
            skipped.append((O,f"error {str(ex)[:120]}")); log(f"  [{ds} assoc] {O}: ERROR {ex}"); LOG.write(traceback.format_exc()+"\n")
    A=pd.DataFrame(assoc)
    if len(A): A=fc.add_fdr(A)
    # write files
    nf=0
    for row in A.itertuples():
        try: _write_assoc(ds,row,meta.get(row.out,{})); nf+=1
        except Exception as ex: log(f"  assoc write err {row.exp}->{row.out}: {ex}")
    if len(A):
        A_out=A.drop(columns=["exp_desc"],errors="ignore")
        A_out.to_csv(os.path.join(SUPPL,f"_manifest_association_{ds}.csv"),index=False,encoding="utf-8-sig")
    log(f"[{ds}] ASSOCIATION done: {nf} files, {len(A)} pairs, {len(skipped)} outcomes skipped")
    return nf, len(A), skipped

def _write_assoc(ds, row, m):
    exp=row.exp; out=row.out; binary=m.get("binary",fc.typ(out)=="b")
    outdir=os.path.join(SUPPL,ds,"association",slug(out)); os.makedirs(outdir,exist_ok=True)
    path=os.path.join(outdir,f"{slug(exp)}__{slug(out)}_{ds}.docx")
    q=row.q; qs="<0.001" if q<0.001 else f"{q:.3f}"
    logged=exp in fc.LOG_EXPOSURES
    per=((" per 1-SD of log" if logged else " per 1-SD") if fc.typ(exp)=="c" else ""); mt="odds ratio" if binary else "beta coefficient"
    pos=(row.est>1) if binary else (row.est>0); direction="positively" if pos else "inversely"
    est=f"{row.est:.2f} (95% CI {row.lo:.2f} to {row.hi:.2f})"
    prev=(f"a weighted {fc.lab(out)} prevalence of {m.get('wprev',float('nan')):.1f} percent, " if binary and not np.isnan(m.get('wprev',float('nan'))) else "")
    abstract=(f"In the {_src(ds)}, {fc.lab(exp)} was {direction} associated with {fc.lab(out)} "
              f"({mt}{per} {est}, false discovery rate q {qs}) among {row.n} adults aged {AGE} years or older "
              f"({prev}survey-weighted). Confounders: {row.adj}. This cross-sectional association is "
              f"hypothesis-generating and requires prospective confirmation.")
    if USE_LLM:
        facts=(f"Survey: {_src(ds)}, survey-weighted, cross-sectional. Exposure: {fc.lab(exp)}. "
               f"Outcome: {fc.lab(out)}. Measure: {mt}{per} {est}. False discovery rate q {qs}. "
               f"Analytic sample {row.n} adults aged {AGE} years or older. "
               f"{('Weighted outcome prevalence ' + prev.strip().rstrip(', ')) if prev else ''} "
               f"Direction: {direction}. Adjustment set: {row.adj}.")
        gen=fc.llm_prose("Write the single-paragraph structured abstract of this analysis. "
                         "End by stating that the association is cross-sectional and "
                         "hypothesis-generating.", facts, LLM_MODEL, LLM_URL)
        if gen: abstract=gen
    doc=Document(); doc.add_heading(f"{fc.lab(exp)} and {fc.lab(out)}",0)
    doc.add_paragraph(f"{ds} · {_src(ds)} · association analysis · {'logistic OR' if binary else 'linear beta'}")
    doc.add_heading("Abstract",1); doc.add_paragraph(abstract)
    doc.add_heading("Methods",1)
    doc.add_paragraph(f"Survey-weighted {'logistic' if binary else 'linear'} regression. {m.get('rat','')} "
                      f"Adjustment for this exposure: {row.adj}. Continuous exposures modeled per 1-SD. "
                      f"{'This exposure was natural-log transformed before standardization owing to a right-skewed distribution. ' if logged else ''}"
                      f"Multiplicity controlled with the Benjamini-Hochberg false discovery rate across the matrix.")
    doc.add_heading("Association",1)
    cols=["Exposure","Outcome","Measure","Adjusted estimate (95% CI)","p","FDR q","N"]
    tb=doc.add_table(rows=1,cols=len(cols)); tb.style="Light Grid Accent 1"
    for j,c in enumerate(cols): tb.rows[0].cells[j].text=c
    ps="<0.001" if (pd.notna(row.p) and row.p<0.001) else (f"{row.p:.3f}" if pd.notna(row.p) else "")
    r0=tb.add_row().cells
    for j,v in enumerate([fc.lab(exp),fc.lab(out),("OR" if binary else "beta"),est,ps,qs,str(row.n)]): r0[j].text=v
    if logged:
        _fn=doc.add_paragraph().add_run(f"Footnote. {fc.lab(exp)} was natural-log transformed before per-1-SD "
                f"standardization owing to its highly right-skewed distribution; the {mt} is per 1-SD increase "
                f"in the natural log of {fc.lab(exp)}.")
        _fn.italic=True
    # descriptive (exposure + confounders by outcome group)
    doc.add_heading("Descriptive (weighted)"+(" by outcome group" if binary else ""),1)
    if binary: dcols=["Variable","Overall","No "+fc.lab(out),fc.lab(out)]
    else: dcols=["Variable","Overall"]
    tb2=doc.add_table(rows=1,cols=len(dcols)); tb2.style="Light Grid Accent 1"
    for j,c in enumerate(dcols): tb2.rows[0].cells[j].text=str(c)
    def addrow(varlabel, trip):
        cc=tb2.add_row().cells; cc[0].text=varlabel
        cc[1].text=str(trip[0])
        if binary: cc[2].text=str(trip[1]); cc[3].text=str(trip[2])
    addrow(fc.lab(exp)+" (exposure)", row.exp_desc)
    for c,trip in m.get("conf_desc",{}).items(): addrow(fc.lab(c), trip)
    doc.add_paragraph("Statistics are deterministic survey-weighted estimates (R survey). "
                      "All findings are hypothesis-generating.")
    doc.save(path)

# ─────────────────────────── TREND (per outcome) ───────────────────────────
def gen_trend(ds, ddir, cycles):
    if len(cycles)<2:
        log(f"[{ds}] TREND skipped ({len(cycles)} cycle) — needs >=2 survey cycles")
        return 0,[("all outcomes","single survey cycle, trend not applicable")]
    log(f"[{ds}] TREND ...")
    outs=[o for o in fc.OUTCOMES if o in fc.AVAIL[ds]]; nf=0; sk=[]
    outdir=os.path.join(SUPPL,ds,"trend"); os.makedirs(outdir,exist_ok=True)
    for oi,O in enumerate(outs):
        try:
            docx=trend_report.build_trend_report(ds, ddir, list(cycles), O, {**DEFS,"pop":{"age_min":max(AGE,19)}},
                                                 LLM_MODEL, LLM_URL, USE_LLM)
            open(os.path.join(outdir,f"{slug(O)}__trend_{ds}.docx"),"wb").write(docx); nf+=1
            log(f"  [{ds} trend {oi+1}/{len(outs)}] {O}: ok")
        except Exception as ex:
            sk.append((O,str(ex)[:120])); log(f"  [{ds} trend] {O}: ERROR {ex}")
    log(f"[{ds}] TREND done: {nf} files, {len(sk)} skipped"); return nf, sk

# ─────────────────────────── ML (per outcome) ───────────────────────────
def gen_ml(ds, DF):
    log(f"[{ds}] ML ...")
    d=fc.apply_definitions(DF,DEFS)
    outs=[o for o in fc.OUTCOMES if o in fc.AVAIL[ds]]; nf=0; sk=[]
    outdir=os.path.join(SUPPL,ds,"ml"); os.makedirs(outdir,exist_ok=True)
    for oi,O in enumerate(outs):
        ev=int((pd.to_numeric(d[O],errors="coerce")==1).sum()) if O in d else 0
        if ev<fc.MIN_EVENTS: sk.append((O,f"low events {ev}")); log(f"  [{ds} ml {oi+1}/{len(outs)}] {O}: SKIP {ev} events"); continue
        try:
            feats=fc.ml_features(ds,O)
            docx=ml_report.build_ml_report(ds, DF, O, feats, DEFS, LLM_MODEL, LLM_URL, USE_LLM,
                                           models=ML_MODELS, sample_n=ML_SAMPLE_N)
            open(os.path.join(outdir,f"{slug(O)}__prediction_{ds}.docx"),"wb").write(docx); nf+=1
            log(f"  [{ds} ml {oi+1}/{len(outs)}] {O}: ok ({len(feats)} features)")
        except Exception as ex:
            sk.append((O,str(ex)[:150])); log(f"  [{ds} ml] {O}: ERROR {ex}"); LOG.write(traceback.format_exc()+"\n")
    log(f"[{ds}] ML done: {nf} files, {len(sk)} skipped"); return nf, sk

def main():
    t0=time.time(); summary=[]
    for ds in ["KNHANES","NHANES"]:
        ddir,cyc=DATA[ds]
        d=fc.apply_definitions(fc.load_raw(ds,ddir,cyc),DEFS)
        a_nf,a_np,a_sk=gen_association(ds,d)
        t_nf,t_sk=gen_trend(ds,ddir,cyc)
        m_nf,m_sk=gen_ml(ds,fc.load_raw(ds,ddir,cyc))
        summary.append((ds,a_nf,a_np,a_sk,t_nf,t_sk,m_nf,m_sk))
    with open(os.path.join(SUPPL,"_SUMMARY.txt"),"w",encoding="utf-8") as f:
        f.write(f"Supplementary generation v2 (association + trend + ML)  elapsed {time.time()-t0:.0f}s\nLocation: {SUPPL}\n")
        f.write(f"Report-writer: {'Llama agent ' + LLM_MODEL if USE_LLM else 'deterministic template (use_llm=False)'}\n")
        if USE_LLM:
            s=fc.LLM_PROSE_STATS
            tot=sum(s.values()) or 1
            f.write(f"  prose accepted from agent : {s['llm']} ({100*s['llm']/tot:.1f}%)\n")
            f.write(f"  numeric-guard violations  : {s['violation']} (deterministic template used)\n")
            f.write(f"  too-short / agent errors  : {s['fallback']} / {s['error']}\n")
        f.write("\n")
        for ds,anf,anp,ask,tnf,tsk,mnf,msk in summary:
            f.write(f"== {ds} ==\n")
            f.write(f"  association: {anf} files ({anp} pairs); outcomes skipped {len(ask)}\n")
            for o,r in ask: f.write(f"      - {fc.lab(o)}: {r}\n")
            f.write(f"  trend: {tnf} files; skipped {len(tsk)}\n")
            for o,r in tsk: f.write(f"      - {fc.lab(o) if o in fc.VARS else o}: {r}\n")
            f.write(f"  ml: {mnf} files; skipped {len(msk)}\n")
            for o,r in msk: f.write(f"      - {fc.lab(o)}: {r}\n")
            f.write("\n")
    log(f"ALL DONE in {time.time()-t0:.0f}s. See {SUPPL}\\_SUMMARY.txt")

if __name__=="__main__": main()
