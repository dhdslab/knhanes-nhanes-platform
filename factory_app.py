# -*- coding: utf-8 -*-
"""KNHANES + NHANES 연구 자동화 플랫폼 (최종본) — Streamlit
인터랙티브 분석 · Trajectory · 역학/트렌드/ML 보고서(Word 자동생성)
실행: streamlit run factory_app.py   (같은 폴더에 factory_core.py, epi_report.py, epi.R, epi_adv.R,
      trend_report.py, trend.R, ml_report.py, engine.R + R survey/rms/MASS + ollama(선택))"""
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
st.set_page_config(page_title="연구 자동화 플랫폼", layout="wide")

with st.sidebar:
    st.header("⚙️ 전역 설정")
    ds=st.radio("데이터셋", fc.DATASETS, horizontal=True)
    DATA_DIRS[ds]=st.text_input("데이터 폴더", DATA_DIRS[ds], key=f"dir_{ds}")
    st.divider(); st.subheader("🤖 LLM (ollama)")
    MODEL=st.text_input("모델",os.getenv("LOCAL_LLM_MODEL","llama3.2:latest")); URL=st.text_input("URL",os.getenv("LOCAL_LLM_URL","http://localhost:11434"))
    USE_LLM=st.checkbox("LLM 사용",value=True)
    st.divider(); AGE=st.number_input("연령 하한",0,100,20)
    st.caption("보고서는 Word(.docx)로 생성됩니다. 분석은 R survey를, ML은 scikit-learn을 사용합니다.")

def get_df():
    k=f"DF_{ds}"
    if k not in st.session_state:
        try:
            st.session_state[k]=fc.load_raw(ds,DATA_DIRS[ds],CYCLES[ds])
        except (FileNotFoundError, RuntimeError) as e:
            st.error(str(e))
            st.info("데이터 파일을 넣은 뒤 사이드바의 데이터 폴더를 맞추고 다시 실행하세요. 예: data/KNHANES")
            st.stop()
    return st.session_state[k]
def defs_(): return {**copy.deepcopy(fc.DEFAULT_DEFS),"pop":{"age_min":AGE}}
def out_avail(): return [o for o in DERIVED if o in fc.AVAIL[ds]]
def dlbtn(docx,fname,key): st.download_button("⬇️ Word(.docx) 다운로드",docx,fname,MIME,key=key,use_container_width=True)

st.title("🏭 KNHANES + NHANES 연구 자동화 플랫폼")
st.caption(f"데이터셋: {ds} · 변수 선택 → 내부 분석 → Word 보고서. 지방간 정의: {fc.STEATOSIS_METHOD[ds]}")
T1,T2,T3,T4,T5=st.tabs(["📊 인터랙티브 분석","📈 Trajectory","🧬 역학 보고서","📉 트렌드 보고서","🤖 ML 보고서"])

# ── 인터랙티브 분석 ──
with T1:
    c1,c2=st.columns(2)
    def vbox(title,dft,kp):
        st.markdown(f"**{title}**"); sel=[]
        for grp,tp in [("연속형","c"),("이분형","b")]:
            st.caption(grp)
            for cvar in [x for x in fc.AVAIL[ds] if fc.typ(x)==tp]:
                if st.checkbox(fc.lab(cvar).split(",")[0],value=cvar in dft,key=f"{kp}_{cvar}"): sel.append(cvar)
        return sel
    with c1: exp=vbox("Exposure",("bodyfat_pct","asm_pct","bmi"),f"{ds}_e")
    with c2: out=vbox("Outcome",("steatosis","dm"),f"{ds}_o")
    cov=fc.auto_covariates(ds,exp,out) if (exp and out) else []
    st.info(f"자동 보정변수: {', '.join(fc.lab(c) for c in cov) or '(선택 시 표시)'}")
    if st.button("▶ 분석 실행",type="primary",key="ia_run",use_container_width=True):
        if not exp or not out: st.error("Exposure/Outcome 최소 1개씩")
        else:
            with st.spinner("설계가중 분석..."):
                d=fc.apply_definitions(get_df(),defs_()); ana=fc.build_analytic(d,exp,out,cov,AGE)
                t1,res,cfg,pairs=fc.run_engine(ana,ds,exp,out,cov); res=fc.add_fdr(res)
                if fc.typ(out[0])=="b":
                    lb=fc.lab(out[0]).split(",")[0]; t1=t1.rename(columns={"0":f"No {lb}","1":lb})
                st.session_state["ia"]=dict(t1=t1,res=res,n=len(ana),exp=exp,out=out,cov=cov)
    if "ia" in st.session_state:
        R=st.session_state["ia"]
        st.markdown(f"#### Table 1 (n={R['n']:,})"); st.dataframe(R["t1"],use_container_width=True,hide_index=True)
        rr=R["res"].copy(); rr["Exposure"]=rr.exposure.str[2:].map(fc.lab); rr["Outcome"]=rr.outcome.map(fc.lab)
        rr["Measure"]=rr.measure.map(fc.measure_name); rr["Estimate (95% CI)"]=rr.apply(lambda x:f"{x.est:.2f} ({x.ci_low:.2f}–{x.ci_high:.2f})",axis=1)
        rr["FDR q"]=rr.q.map(lambda q:"<0.001" if q<0.001 else f"{q:.3f}")
        st.markdown("#### 연관성 (이분형=OR, 연속형=β)"); st.dataframe(rr[["Exposure","Outcome","Measure","Estimate (95% CI)","FDR q"]].sort_values("FDR q"),use_container_width=True,hide_index=True)
        if st.button("📝 논문 Word 생성",key="ia_ms",use_container_width=True):
            with st.spinner("작성 중..."):
                ms=fc.gen_manuscript(R["t1"],R["res"],ds,R["exp"],R["out"],R["cov"],defs_(),MODEL,URL,USE_LLM)
                docx=fc.build_docx(f"Association study ({ds})",ms,R["t1"],R["res"])
            dlbtn(docx,f"manuscript_{ds}.docx","ia_dl")

# ── Trajectory ──
with T2:
    st.markdown("outcome 결정변수의 연령별 설계가중 평균 (+ 유병률)")
    o=st.selectbox("Outcome",out_avail(),format_func=fc.lab,key="tj_o"); sx=st.checkbox("성별 분리",True,key="tj_s")
    if st.button("📈 Trajectory 그리기",key="tj_b",use_container_width=True):
        with st.spinner("계산 중..."):
            d=fc.apply_definitions(get_df(),defs_()); tdf,vs=fc.trajectory(d,ds,o,split_sex=sx)
        if not vs: st.warning("결정 변수가 현재 데이터에 없습니다.")
        else: st.caption(f"결정변수: {vs}"); st.pyplot(fc.plot_trajectory(tdf,o,ds,sx))

# ── 역학 보고서 ──
with T3:
    st.markdown("outcome·주요 노출 선택 → Table 1~6(+SMD·crude/adj OR·하위군·IPTW/AIPW/G-comp·Love·VIF·RCS·E-value) Word")
    c1,c2=st.columns(2)
    o=c1.selectbox("Outcome (이분형)",out_avail(),format_func=fc.lab,key="epi_o")
    mx=c2.selectbox("주요 노출",[v for v in fc.AVAIL[ds] if v!=o],format_func=fc.lab,key="epi_x")
    cov=fc.auto_covariates(ds,[mx],[o]); st.info(f"자동 보정: {', '.join(fc.lab(c) for c in cov)}")
    if st.button("🧬 역학 보고서 생성 (Word)",type="primary",key="epi_b",use_container_width=True):
        with st.spinner("역학 분석(설계가중 + 인과추론)... 수십 초 소요"):
            try:
                docx=epi_report.build_epi_report(ds,get_df(),o,mx,cov,["men","age50"],defs_(),MODEL,URL,USE_LLM)
                st.success("역학 보고서 생성 완료"); dlbtn(docx,f"역학분석_{ds}_{o}.docx","epi_dl")
            except Exception as e: st.error(f"오류: {e}")

# ── 트렌드 보고서 ──
with T4:
    st.markdown("연도별 유병률·age-sex 표준화·Projection·층화·Joinpoint APC·음이항 예측 Word")
    o=st.selectbox("Outcome",out_avail(),format_func=fc.lab,key="tr_o")
    yrs=st.multiselect("연도(사이클)",TREND_YEARS[ds],default=TREND_YEARS[ds],key="tr_y")
    if st.button("📉 트렌드 보고서 생성 (Word)",type="primary",key="tr_b",use_container_width=True):
        if len(yrs)<2: st.error("최소 2개 연도를 선택하세요.")
        else:
            with st.spinner("추세 분석(표준화·APC 부트스트랩·NB 예측)..."):
                try:
                    docx=trend_report.build_trend_report(ds,DATA_DIRS[ds],yrs,o,{**defs_(),"pop":{"age_min":max(AGE,19)}},MODEL,URL,USE_LLM)
                    st.success("트렌드 보고서 생성 완료"); dlbtn(docx,f"트렌드_{ds}_{o}.docx","tr_dl")
                except Exception as e: st.error(f"오류: {e}")

# ── ML 보고서 ──
with T5:
    st.markdown("예측모형 — 튜닝·탐색공간·전체지표·ROC/PR·threshold·calibration·SHAP·PDP Word")
    o=st.selectbox("Outcome (이분형)",out_avail(),format_func=fc.lab,key="ml_o")
    preds=st.multiselect("예측변수(피처)",[v for v in fc.AVAIL[ds] if v!=o],
        default=[v for v in ["age","men","bmi","wc","sbp","tg","alt","bodyfat_pct","asm_pct","smoking","alcohol"] if v in fc.AVAIL[ds] and v!=o],
        format_func=fc.lab,key="ml_p")
    st.caption("주의: 결과의 정의 성분(예: 당뇨↔혈당)을 피처에 넣으면 정보누출이 됩니다. 예측이 목적이면 제외하세요.")
    if st.button("🤖 ML 보고서 생성 (Word)",type="primary",key="ml_b",use_container_width=True):
        if len(preds)<2: st.error("피처를 2개 이상 선택하세요.")
        else:
            with st.spinner("모델 튜닝·평가·SHAP... 1~2분 소요"):
                try:
                    docx=ml_report.build_ml_report(ds,get_df(),o,preds,defs_(),MODEL,URL,USE_LLM)
                    st.success("ML 보고서 생성 완료"); dlbtn(docx,f"ML_{ds}_{o}.docx","ml_dl")
                except Exception as e: st.error(f"오류: {e}")
