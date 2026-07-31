# -*- coding: utf-8 -*-
"""KNHANES + NHANES research automation platform (final) -- Streamlit
Interactive analysis, Trajectory, epidemiology/trend/ML reports (auto-generated Word)
Run: streamlit run factory_app.py   (keep factory_core.py, epi_report.py, epi.R, epi_adv.R,
      trend_report.py, trend.R, ml_report.py, engine.R in the same folder + R survey/rms/MASS + ollama (optional))"""
import streamlit as st
import factory_core as fc
import epi_report, trend_report, ml_report
import copy
import os
MIME="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DERIVED=["steatosis","dm","htn","mets","dld"]
DATA_DIRS={"KNHANES":"data/KNHANES","NHANES":"data/NHANES"}
CYCLES={"KNHANES":("08","09"),"NHANES":("j",)}
TREND_YEARS={"KNHANES":["08","09","10"],"NHANES":["j"]}
st.set_page_config(page_title="Research Automation Platform", layout="wide")

with st.sidebar:
    st.header("Global settings")
    ds=st.radio("Dataset", fc.DATASETS, horizontal=True)
    DATA_DIRS[ds]=st.text_input("Data folder", DATA_DIRS[ds], key=f"dir_{ds}")
    st.divider(); st.subheader("LLM (ollama)")
    MODEL=st.text_input("Model",os.getenv("LOCAL_LLM_MODEL","llama3.2:latest")); URL=st.text_input("URL",os.getenv("LOCAL_LLM_URL","http://localhost:11434"))
    USE_LLM=st.checkbox("Use LLM",value=True)
    st.divider(); AGE=st.number_input("Minimum age",0,100,20)
    st.caption("Reports are generated as Word (.docx). Analysis uses R survey; ML uses scikit-learn.")

def get_df():
    k=f"DF_{ds}"
    if k not in st.session_state:
        try:
            st.session_state[k]=fc.load_raw(ds,DATA_DIRS[ds],CYCLES[ds])
        except (FileNotFoundError, RuntimeError) as e:
            st.error(str(e))
            st.info("Place the data files, set the data folder in the sidebar, and run again. Example: data/KNHANES")
            st.stop()
    return st.session_state[k]
def defs_(): return {**copy.deepcopy(fc.DEFAULT_DEFS),"pop":{"age_min":AGE}}
def out_avail(): return [o for o in DERIVED if o in fc.AVAIL[ds]]
def dlbtn(docx,fname,key): st.download_button("Download Word (.docx)",docx,fname,MIME,key=key,use_container_width=True)

st.title("KNHANES + NHANES Research Automation Platform")
st.caption(f"Dataset: {ds} - variable selection -> internal analysis -> Word report. Steatosis definition: {fc.STEATOSIS_METHOD[ds]}")
T1,T2,T3,T4,T5=st.tabs(["Interactive analysis","Trajectory","Epidemiology report","Trend report","ML report"])

# -- Interactive analysis --
with T1:
    c1,c2=st.columns(2)
    def vbox(title,dft,kp):
        st.markdown(f"**{title}**"); sel=[]
        for grp,tp in [("Continuous","c"),("Binary","b")]:
            st.caption(grp)
            for cvar in [x for x in fc.AVAIL[ds] if fc.typ(x)==tp]:
                if st.checkbox(fc.lab(cvar).split(",")[0],value=cvar in dft,key=f"{kp}_{cvar}"): sel.append(cvar)
        return sel
    with c1: exp=vbox("Exposure",("bodyfat_pct","asm_pct","bmi"),f"{ds}_e")
    with c2: out=vbox("Outcome",("steatosis","dm"),f"{ds}_o")
    cov=fc.auto_covariates(ds,exp,out) if (exp and out) else []
    st.info(f"Auto-selected covariates: {', '.join(fc.lab(c) for c in cov) or '(shown after selection)'}")
    if st.button("Run analysis",type="primary",key="ia_run",use_container_width=True):
        if not exp or not out: st.error("Select at least one Exposure and one Outcome")
        else:
            with st.spinner("Survey-weighted analysis..."):
                d=fc.apply_definitions(get_df(),defs_()); ana=fc.build_analytic(d,exp,out,cov,AGE)
                t1,res,cfg,pairs=fc.run_engine(ana,ds,exp,out,cov); res=fc.add_fdr(res)
                if fc.typ(out[0])=="b":
                    lb=fc.lab(out[0]).split(",")[0]; t1=t1.rename(columns={"0":f"No {lb}","1":lb})
                st.session_state["ia"]=dict(t1=t1,res=res,n=len(ana),exp=exp,out=out,cov=cov)
    if "ia" in st.session_state:
        R=st.session_state["ia"]
        st.markdown(f"#### Table 1 (n={R['n']:,})"); st.dataframe(R["t1"],use_container_width=True,hide_index=True)
        rr=R["res"].copy(); rr["Exposure"]=rr.exposure.str[2:].map(fc.lab); rr["Outcome"]=rr.outcome.map(fc.lab)
        rr["Measure"]=rr.measure.map(fc.measure_name); rr["Estimate (95% CI)"]=rr.apply(lambda x:f"{x.est:.2f} ({x.ci_low:.2f}-{x.ci_high:.2f})",axis=1)
        rr["FDR q"]=rr.q.map(lambda q:"<0.001" if q<0.001 else f"{q:.3f}")
        st.markdown("#### Associations (binary=OR, continuous=beta)"); st.dataframe(rr[["Exposure","Outcome","Measure","Estimate (95% CI)","FDR q"]].sort_values("FDR q"),use_container_width=True,hide_index=True)
        if st.button("Generate manuscript (Word)",key="ia_ms",use_container_width=True):
            with st.spinner("Writing..."):
                ms=fc.gen_manuscript(R["t1"],R["res"],ds,R["exp"],R["out"],R["cov"],defs_(),MODEL,URL,USE_LLM)
                docx=fc.build_docx(f"Association study ({ds})",ms,R["t1"],R["res"])
            dlbtn(docx,f"manuscript_{ds}.docx","ia_dl")

# -- Trajectory --
with T2:
    st.markdown("Age-specific survey-weighted means of outcome determinants (+ prevalence)")
    o=st.selectbox("Outcome",out_avail(),format_func=fc.lab,key="tj_o"); sx=st.checkbox("Split by sex",True,key="tj_s")
    if st.button("Plot trajectory",key="tj_b",use_container_width=True):
        with st.spinner("Computing..."):
            d=fc.apply_definitions(get_df(),defs_()); tdf,vs=fc.trajectory(d,ds,o,split_sex=sx)
        if not vs: st.warning("Determinant variables are not present in the current data.")
        else: st.caption(f"Determinants: {vs}"); st.pyplot(fc.plot_trajectory(tdf,o,ds,sx))

# -- Epidemiology report --
with T3:
    st.markdown("Select outcome and main exposure -> Table 1-6 (+SMD, crude/adj OR, subgroups, IPTW/AIPW/G-comp, Love, VIF, RCS, E-value) Word")
    c1,c2=st.columns(2)
    o=c1.selectbox("Outcome (binary)",out_avail(),format_func=fc.lab,key="epi_o")
    mx=c2.selectbox("Main exposure",[v for v in fc.AVAIL[ds] if v!=o],format_func=fc.lab,key="epi_x")
    cov=fc.auto_covariates(ds,[mx],[o]); st.info(f"Auto-adjusted: {', '.join(fc.lab(c) for c in cov)}")
    if st.button("Generate epidemiology report (Word)",type="primary",key="epi_b",use_container_width=True):
        with st.spinner("Epidemiologic analysis (survey-weighted + causal inference)... tens of seconds"):
            try:
                docx=epi_report.build_epi_report(ds,get_df(),o,mx,cov,["men","age50"],defs_(),MODEL,URL,USE_LLM)
                st.success("Epidemiology report generated"); dlbtn(docx,f"epi_{ds}_{o}.docx","epi_dl")
            except Exception as e: st.error(f"Error: {e}")

# -- Trend report --
with T4:
    st.markdown("Yearly prevalence, age-sex standardization, projection, stratification, Joinpoint APC, negative-binomial forecast Word")
    o=st.selectbox("Outcome",out_avail(),format_func=fc.lab,key="tr_o")
    yrs=st.multiselect("Years (cycles)",TREND_YEARS[ds],default=TREND_YEARS[ds],key="tr_y")
    if st.button("Generate trend report (Word)",type="primary",key="tr_b",use_container_width=True):
        if len(yrs)<2: st.error("Select at least 2 years.")
        else:
            with st.spinner("Trend analysis (standardization, APC bootstrap, NB forecast)..."):
                try:
                    docx=trend_report.build_trend_report(ds,DATA_DIRS[ds],yrs,o,{**defs_(),"pop":{"age_min":max(AGE,19)}},MODEL,URL,USE_LLM)
                    st.success("Trend report generated"); dlbtn(docx,f"trend_{ds}_{o}.docx","tr_dl")
                except Exception as e: st.error(f"Error: {e}")

# -- ML report --
with T5:
    st.markdown("Prediction models -- tuning, search space, full metrics, ROC/PR, threshold, calibration, SHAP, PDP Word")
    o=st.selectbox("Outcome (binary)",out_avail(),format_func=fc.lab,key="ml_o")
    preds=st.multiselect("Predictors (features)",[v for v in fc.AVAIL[ds] if v!=o],
        default=[v for v in ["age","men","bmi","wc","sbp","tg","alt","bodyfat_pct","asm_pct","smoking","alcohol"] if v in fc.AVAIL[ds] and v!=o],
        format_func=fc.lab,key="ml_p")
    st.caption("Note: putting an outcome's definitional components (e.g., diabetes <-> glucose) into the features causes information leakage. Exclude them if prediction is the goal.")
    if st.button("Generate ML report (Word)",type="primary",key="ml_b",use_container_width=True):
        if len(preds)<2: st.error("Select 2 or more features.")
        else:
            with st.spinner("Model tuning, evaluation, SHAP... 1-2 minutes"):
                try:
                    docx=ml_report.build_ml_report(ds,get_df(),o,preds,defs_(),MODEL,URL,USE_LLM)
                    st.success("ML report generated"); dlbtn(docx,f"ml_{ds}_{o}.docx","ml_dl")
                except Exception as e: st.error(f"Error: {e}")
