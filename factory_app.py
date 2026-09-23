# -*- coding: utf-8 -*-
"""KNHANES + NHANES Research Automation Platform (final v4) — Streamlit

Harmonized outcomes definable identically in BOTH surveys, plus country-specific outcomes
(COPD in KNHANES, MDD/PHQ-9 in NHANES). All exposures x all outcomes. Smoking/alcohol are binary
never vs past/current. Outcomes with too few events are blocked. Categorical binnings are identical
across surveys. Reports auto-generate as Word (.docx).

Run: streamlit run factory_app.py   (same folder: factory_core.py, epi_report.py, epi.R, epi_adv.R,
     epi_surv.R, trend_report.py, trend.R, ml_report.py, engine.R + R survey/splines/MASS + ollama optional)
"""
import streamlit as st
import factory_core as fc
import epi_report, trend_report, ml_report, full_report
import pandas as pd, copy
MIME="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DATA_DIRS={"KNHANES":"data/KNHANES","NHANES":"data/NHANES"}
_KN_CYC=("07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24")
CYCLES={"KNHANES":_KN_CYC,"NHANES":("d","e","f","g","h","i","j","l")}
TREND_YEARS={"KNHANES":list(_KN_CYC),"NHANES":["d","e","f","g","h","i","j","l"]}
st.set_page_config(page_title="Research Automation Platform", layout="wide")

with st.sidebar:
    st.header("⚙️ Global settings")
    ds=st.radio("Dataset", fc.DATASETS, horizontal=True)
    DATA_DIRS[ds]=st.text_input("Data folder", DATA_DIRS[ds], key=f"dir_{ds}")
    st.divider(); st.subheader("🤖 LLM (ollama)")
    MODEL=st.text_input("Model","llama3.3:70b"); URL=st.text_input("URL","http://localhost:11434")
    USE_LLM=st.checkbox("Use LLM (ollama) to write prose",value=True)
    st.caption("On = llama writes report/manuscript text from the computed numbers (deterministic fallback if "
               "ollama is down). Statistics are always R survey, never from the LLM.")
    st.divider(); AGE=st.number_input("Minimum age",0,100,20)
    st.caption(f"Event floor: outcomes with < {fc.MIN_EVENTS} events are blocked.")

def get_df():
    k=f"DF_{ds}"
    if k not in st.session_state: st.session_state[k]=fc.load_raw(ds,DATA_DIRS[ds],CYCLES[ds])
    return st.session_state[k]
def get_derived():
    k=f"DER_{ds}_{AGE}"
    if k not in st.session_state: st.session_state[k]=fc.apply_definitions(get_df(),defs_())
    return st.session_state[k]
def defs_(): return {**copy.deepcopy(fc.DEFAULT_DEFS),"pop":{"age_min":AGE}}
def avail_out(include_cont=False):
    outs=[o for o in fc.OUTCOMES if o in fc.AVAIL[ds]]
    if include_cont: outs=outs+[v for v in fc.AVAIL[ds] if fc.typ(v)=="c" and v not in outs]
    return outs
def event_ok(d, outcomes):
    """req #8: return (ok, message) checking every BINARY outcome clears the event floor.
    Continuous outcomes (linear beta) have no event floor."""
    low=[]
    for o in outcomes:
        if o in d.columns and fc.typ(o)=="b":
            n=int((pd.to_numeric(d[o],errors="coerce")==1).sum())
            if n<fc.MIN_EVENTS: low.append(f"{fc.lab(o)} ({n})")
    return (len(low)==0, "; ".join(low))
def dlbtn(docx,fname,key): st.download_button("⬇️ Download Word (.docx)",docx,fname,MIME,key=key,use_container_width=True)

st.title("🏭 KNHANES + NHANES Research Automation Platform")
st.caption(f"Dataset: {ds} · select variables → survey-weighted analysis → Word report. "
           f"Steatosis: {fc.STEATOSIS_METHOD[ds]}")
st.info("Common outcomes are defined **identically in both surveys**. Country-specific: "
        "**COPD** is KNHANES-only (no NHANES spirometry), **MDD (PHQ-9)** is NHANES-only "
        "(no PHQ-9 in KNHANES 2008–2011).")
T1,T2,T3,T4,T5,T6,T7=st.tabs(["📊 Interactive analysis","📈 Trajectory","🧬 Epidemiologic report",
                              "📉 Trend report","🤖 ML report","🧾 Auto-manuscript (all pairs)",
                              "📚 Combined report (all-in-one)"])

# ── Interactive analysis (any variable as exposure or outcome; all-pairs) ──
with T1:
    st.markdown("Any variable can be an **exposure or an outcome** (roles are symmetric). Binary outcome "
                "→ logistic OR, continuous outcome → linear β. Smoking/alcohol are binary never vs past/current. "
                "Definitionally circular pairs are skipped automatically.")
    UNI=fc.exposure_candidates(ds)   # full variable universe (same set on both sides)
    def rolebox(title, key, defaults):
        st.markdown(f"**{title}**")
        allsel=st.checkbox(f"Select all", key=f"all_{key}")
        picked=[]
        for grp,tp in [("Continuous","c"),("Binary","b")]:
            st.caption(grp)
            for cvar in [x for x in UNI if fc.typ(x)==tp]:
                dft=allsel or cvar in defaults
                if st.checkbox(fc.lab(cvar).split(",")[0], value=dft, key=f"{ds}_{key}_{cvar}"): picked.append(cvar)
        return picked
    c1,c2=st.columns(2)
    with c1: exp=rolebox("Exposures", "e", ("bodyfat_pct","asm_pct","bmi"))
    with c2: out=rolebox("Outcomes", "o", ("dm","masld"))
    cov=fc.auto_covariates(ds,exp,out) if (exp and out) else []
    st.info(f"Auto covariates (over-adjustment avoided): {', '.join(fc.lab(c) for c in cov) or '(shown after selection)'}")
    if st.button("▶ Run analysis",type="primary",key="ia_run",use_container_width=True):
        if not exp or not out: st.error("Select at least one exposure and one outcome")
        else:
            with st.spinner("Survey-weighted analysis (R survey)..."):
                d=get_derived()
                ok,low=event_ok(d,out)
                if not ok: st.error(f"Blocked — outcomes below the {fc.MIN_EVENTS}-event floor: {low}")
                else:
                    ana=fc.build_analytic(d,exp,out,cov,AGE)
                    t1,res,cfg,pairs=fc.run_engine(ana,ds,exp,out,cov); res=fc.add_fdr(res)
                    if fc.typ(out[0])=="b":
                        lb=fc.lab(out[0]).split(",")[0]; t1=t1.rename(columns={"0":f"No {lb}","1":lb})
                    st.session_state["ia"]=dict(t1=t1,res=res,n=len(ana),exp=exp,out=out,cov=cov,
                                                ev=fc.event_counts(ana,out))
    if "ia" in st.session_state:
        R=st.session_state["ia"]
        st.caption("Events per outcome: "+", ".join(f"{fc.lab(o)}={n}" for o,n in R["ev"].items()))
        st.markdown(f"#### Table 1 (n={R['n']:,})"); st.dataframe(R["t1"],use_container_width=True,hide_index=True)
        rr=R["res"].copy(); rr["Exposure"]=rr.exposure.str[2:].map(fc.lab); rr["Outcome"]=rr.outcome.map(fc.lab)
        rr["Measure"]=rr.measure.map(fc.measure_name)
        rr["Estimate (95% CI)"]=rr.apply(lambda x:f"{x.est:.2f} ({x.ci_low:.2f}–{x.ci_high:.2f})",axis=1)
        rr["FDR q"]=rr.q.map(lambda q:"<0.001" if q<0.001 else f"{q:.3f}")
        st.markdown("#### Associations (binary outcome = OR, continuous = β)")
        st.dataframe(rr[["Exposure","Outcome","Measure","Estimate (95% CI)","FDR q"]].sort_values("FDR q"),
                     use_container_width=True,hide_index=True)
        if st.button("📝 Generate manuscript Word",key="ia_ms",use_container_width=True):
            with st.spinner("Writing..."):
                ms=fc.gen_manuscript(R["t1"],R["res"],ds,R["exp"],R["out"],R["cov"],defs_(),MODEL,URL,USE_LLM)
                docx=fc.build_docx(f"Association study ({ds})",ms,R["t1"],R["res"])
            dlbtn(docx,f"manuscript_{ds}.docx","ia_dl")

# ── Trajectory ──
with T2:
    st.markdown("Age-specific survey-weighted means of outcome determinants (+ prevalence)")
    o=st.selectbox("Outcome",avail_out(),format_func=fc.lab,key="tj_o"); sx=st.checkbox("Split by sex",True,key="tj_s")
    if st.button("📈 Plot trajectory",key="tj_b",use_container_width=True):
        with st.spinner("Computing..."):
            d=get_derived(); tdf,vs=fc.trajectory(d,ds,o,split_sex=sx)
        if not vs: st.warning("Determinant variables are not available in this dataset.")
        else: st.caption(f"Determinants: {[fc.lab(v) for v in vs]}"); st.pyplot(fc.plot_trajectory(tdf,o,ds,sx))

# ── Epidemiologic report ──
with T3:
    st.markdown("Outcome + main exposure → Table 1–6 (SMD, crude/adj OR, subgroup+P-int, "
                "PSM·IPTW·AIPW·G-comp·TMLE, Love, VIF, RCS, E-value) Word")
    c1,c2=st.columns(2)
    o=c1.selectbox("Outcome (binary)",avail_out(),format_func=fc.lab,key="epi_o")
    mx=c2.selectbox("Main exposure",[v for v in fc.AVAIL[ds] if not fc.related(ds,v,o)],
                    format_func=fc.lab,key="epi_x")
    cov=fc.auto_covariates(ds,[mx],[o]); st.info(f"Auto covariates: {', '.join(fc.lab(c) for c in cov)}")
    if st.button("🧬 Generate epidemiologic report (Word)",type="primary",key="epi_b",use_container_width=True):
        d=get_derived(); ok,low=event_ok(d,[o])
        if not ok: st.error(f"Blocked — {fc.lab(o)} has fewer than {fc.MIN_EVENTS} events ({low}).")
        else:
            with st.spinner("Epidemiologic analysis (survey-weighted + causal inference)... tens of seconds"):
                try:
                    docx=epi_report.build_epi_report(ds,get_df(),o,mx,cov,["men","age50"],defs_(),MODEL,URL,USE_LLM)
                    st.success("Epidemiologic report generated"); dlbtn(docx,f"epi_{ds}_{o}.docx","epi_dl")
                except Exception as e: st.error(f"Error: {e}")

# ── Trend report ──
with T4:
    st.markdown("Yearly prevalence, age-sex standardization, projection, stratification, joinpoint APC, NB forecast Word")
    if len(TREND_YEARS[ds])<2:
        st.warning(f"{ds} has a single cycle here; trends need ≥2 cycles. Add more NHANES cycles to enable trends.")
    o=st.selectbox("Outcome",avail_out(),format_func=fc.lab,key="tr_o")
    yrs=st.multiselect("Years (cycles)",TREND_YEARS[ds],default=TREND_YEARS[ds],key="tr_y")
    if st.button("📉 Generate trend report (Word)",type="primary",key="tr_b",use_container_width=True):
        if len(yrs)<2: st.error("Select at least 2 years.")
        else:
            with st.spinner("Trend analysis (standardization, APC bootstrap, NB forecast)..."):
                try:
                    docx=trend_report.build_trend_report(ds,DATA_DIRS[ds],yrs,o,{**defs_(),"pop":{"age_min":max(AGE,19)}},MODEL,URL,USE_LLM)
                    st.success("Trend report generated"); dlbtn(docx,f"trend_{ds}_{o}.docx","tr_dl")
                except Exception as e: st.error(f"Error: {e}")

# ── ML report ──
with T5:
    st.markdown("Prediction model — tuning, full metrics, ROC/PR, threshold, calibration, SHAP, PDP Word")
    o=st.selectbox("Outcome (binary)",avail_out(),format_func=fc.lab,key="ml_o")
    availp=[v for v in fc.AVAIL[ds] if not fc.related(ds,v,o)]
    preds=st.multiselect("Predictors (features)",availp,
        default=[v for v in ["age","men","bmi","wc","sbp","tg","alt","bodyfat_pct","asm_pct","smoking","alcohol"] if v in availp],
        format_func=fc.lab,key=f"ml_p_{o}")
    st.caption("Leakage prevention: definitional components of "+fc.lab(o)+" are auto-excluded"
               +(f" ({', '.join(fc.lab(x) for x in fc.determinants(ds,o))})" if fc.determinants(ds,o) else "")+".")
    mdls=st.multiselect("Models to compare",ml_report.ALL_MODELS,default=ml_report.DEFAULT_MODELS,key="ml_m")
    st.caption("SVM-RBF and MLP are excluded by default (slow) but selectable. The other 14 are default.")
    if st.button("🤖 Generate ML report (Word)",type="primary",key="ml_b",use_container_width=True):
        if len(preds)<2 or len(mdls)<2: st.error("Select at least 2 features and 2 models.")
        else:
            d=get_derived(); ok,low=event_ok(d,[o])
            if not ok: st.error(f"Blocked — {fc.lab(o)} has fewer than {fc.MIN_EVENTS} events ({low}).")
            else:
                with st.spinner(f"Tuning/evaluating {len(mdls)} models + SHAP... a few minutes"):
                    try:
                        docx=ml_report.build_ml_report(ds,get_df(),o,preds,defs_(),MODEL,URL,USE_LLM,models=mdls)
                        st.success("ML report generated"); dlbtn(docx,f"ML_{ds}_{o}.docx","ml_dl")
                    except Exception as e: st.error(f"Error: {e}")

# ── Auto-manuscript: every exposure × every outcome, then llama writes the paper ──
with T6:
    st.markdown("Run **every exposure × every available outcome** (each outcome in its own complete-data "
                "sample), then **llama writes a full manuscript** (Abstract to Conclusion) from the computed "
                "numbers. Output: Word (.docx). Roles are symmetric: any variable can be an exposure or an "
                "outcome (binary → OR, continuous → β).")
    UNI=fc.exposure_candidates(ds); OUT_UNI=fc.outcome_candidates(ds)
    all_e2=st.checkbox("Use all available exposures",value=True,key="am_all")
    default_e=[v for v in ["age","men","bmi","wc","sbp","tg","alt","bodyfat_pct","asm_pct","smoking","alcohol"] if v in UNI]
    exps=UNI[:] if all_e2 else st.multiselect("Exposures",UNI,default=default_e,format_func=fc.lab,key="am_exp")
    all_o2=st.checkbox("Use all clinical binary outcomes",value=True,key="am_allo")
    default_o=[o for o in fc.OUTCOMES if o in fc.AVAIL[ds]]
    outs=default_o if all_o2 else st.multiselect("Outcomes (any variable)",OUT_UNI,default=default_o,format_func=fc.lab,key="am_out")
    st.caption(f"This will estimate up to {len(exps)}×{len(outs)} associations; covariates and leakage/over-adjustment "
               "are handled automatically per outcome.")
    if st.button("▶ Run all pairs",type="primary",key="am_run",use_container_width=True):
        if len(exps)<1: st.error("Select at least one exposure.")
        else:
            d=get_derived(); prog=st.progress(0.0,text="Starting...")
            def cb(i,n,o): prog.progress(i/max(n,1),text=f"Outcome {i+1}/{n}: {fc.lab(o)}")
            with st.spinner("Survey-weighted analysis over all pairs (R survey)..."):
                RES,info,t1=fc.run_matrix(d,exps,outs,ds,AGE,progress=cb)
            prog.progress(1.0,text="Analysis complete")
            st.session_state["am"]=dict(RES=RES,info=info,t1=t1,exps=exps,outs=outs)
    if "am" in st.session_state:
        A=st.session_state["am"]
        st.markdown("#### Per-outcome status"); st.dataframe(A["info"],use_container_width=True,hide_index=True)
        if len(A["RES"]):
            rr=A["RES"].copy(); rr["Exposure"]=rr.exposure.str[2:].map(fc.lab); rr["Outcome"]=rr.outcome.map(fc.lab)
            rr["Measure"]=rr.measure.map(fc.measure_name)
            rr["Estimate (95% CI)"]=rr.apply(lambda x:f"{x.est:.2f} ({x.ci_low:.2f}–{x.ci_high:.2f})",axis=1)
            rr["FDR q"]=rr.q.map(lambda q:"<0.001" if q<0.001 else f"{q:.3f}")
            nsig=int((A["RES"].q<0.05).sum())
            st.markdown(f"#### Associations — {len(A['RES'])} pairs, {nsig} significant (FDR q<0.05)")
            st.dataframe(rr[["Exposure","Outcome","Measure","Estimate (95% CI)","FDR q"]].sort_values("FDR q"),
                         use_container_width=True,hide_index=True)
            if st.button("📝 Write full manuscript with llama (Word)",key="am_ms",type="primary",use_container_width=True):
                with st.spinner("llama is writing the manuscript from the computed numbers..."):
                    secs=fc.gen_full_manuscript(A["RES"],A["info"],ds,A["exps"],A["outs"],MODEL,URL,USE_LLM)
                    docx=fc.build_full_manuscript_docx(A["RES"],A["info"],secs,ds,A["exps"],A["outs"],A["t1"])
                st.success(f"Manuscript generated (prose source: {secs.get('_source')})")
                dlbtn(docx,f"manuscript_full_{ds}.docx","am_dl")
        else:
            st.warning("No associations were estimated (check event floor / exposure selection).")

# ── Combined all-in-one report: epidemiology + trend + ML → one paper ──
with T7:
    st.markdown("Run the **epidemiologic, trend, and machine-learning** analyses for one exposure/outcome and "
                "assemble them into **one paper-format Word file** (Abstract · Introduction · Part I epidemiology · "
                "Part II trends · Part III prediction · Discussion · Conclusion). Takes a couple of minutes.")
    c1,c2=st.columns(2)
    o=c1.selectbox("Outcome (binary)",avail_out(),format_func=fc.lab,key="fr_o")
    mx=c2.selectbox("Main exposure",[v for v in fc.AVAIL[ds] if not fc.related(ds,v,o)],format_func=fc.lab,key="fr_x")
    yrs=st.multiselect("Trend years (cycles)",TREND_YEARS[ds],default=TREND_YEARS[ds],key="fr_y")
    fast=[m for m in ["LogisticRegression","RandomForest","GradientBoosting","XGBoost","HistGradientBoosting"] if m in ml_report.ALL_MODELS]
    mdls=st.multiselect("ML models",ml_report.ALL_MODELS,default=fast,key="fr_m")
    st.caption("Covariates and ML features are auto-selected (leakage and over-adjustment avoided). "
               "Trend needs ≥2 cycles; otherwise that part is noted as unavailable.")
    if st.button("📚 Generate combined report (Word)",type="primary",key="fr_b",use_container_width=True):
        d=get_derived(); ok,low=event_ok(d,[o])
        if not ok: st.error(f"Blocked — {fc.lab(o)} has fewer than {fc.MIN_EVENTS} events ({low}).")
        elif len(mdls)<2: st.error("Select at least 2 ML models.")
        else:
            with st.spinner("Running epidemiology + trend + ML and assembling the paper... a couple of minutes"):
                try:
                    docx=full_report.build_full_report(ds,get_df(),DATA_DIRS[ds],o,mx,defs_(),MODEL,URL,USE_LLM,
                                                       years=yrs,models=mdls)
                    st.success("Combined report generated")
                    dlbtn(docx,f"combined_{ds}_{mx}_{o}.docx","fr_dl")
                except Exception as e: st.error(f"Error: {e}")
