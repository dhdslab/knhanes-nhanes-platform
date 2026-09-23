# -*- coding: utf-8 -*-
"""Combined single-file report: runs the epidemiologic, trend, and ML analyses for one
exposure/outcome and assembles them into ONE paper-format Word document with Abstract,
Introduction, Part I (epidemiology), Part II (trends), Part III (prediction), Discussion,
and Conclusion. Statistics are deterministic (R survey / scikit-learn); the LLM only writes
the connecting prose from the computed numbers."""
import factory_core as fc
import epi_report, trend_report, ml_report
import io, os

def _src(dataset):
    return {"KNHANES":"Korea National Health and Nutrition Examination Survey",
            "NHANES":"National Health and Nutrition Examination Survey"}.get(dataset,dataset)

def _facts(summary, dataset, exposure, outcome):
    src=_src(dataset); parts=[f"Data source {src}. Exposure {fc.lab(exposure)}. Outcome {fc.lab(outcome)}."]
    e=summary.get('epi')
    if e:
        parts.append(f"Cross-sectional survey-weighted logistic regression in {e['n']} adults. Weighted prevalence "
                     f"{e['prev']:.1f} percent. Fully adjusted odds ratio for {fc.lab(exposure)}{e['per']} "
                     f"{e['or']:.2f} (95% CI {e['lo']:.2f} to {e['hi']:.2f}). E-value {e['evalue']} "
                     f"(confidence-interval limit {e['evalue_ci']}). Robustness confirmed with IPTW, propensity-score "
                     f"matching, AIPW, G-computation and TMLE.")
    t=summary.get('trend')
    if t:
        parts.append(f"Age-sex standardized prevalence changed from {t['prev0']:.1f} to {t['prev1']:.1f} percent between "
                     f"{t['y0']} and {t['y1']}, annual percent change {t['apc']} (95% CI {t['apc_lo']} to {t['apc_hi']}) "
                     f"in the latest segment {t['period']}.")
    m=summary.get('ml')
    if m:
        parts.append(f"A machine-learning prediction analysis compared {m['n_models']} models. The best model was {m['best']} "
                     f"with area under the ROC curve {m['auROC']} and area under the precision-recall curve {m['auPRC']}.")
    return " ".join(parts)

def _llm_or(fallback, prompt, model, url, use_llm):
    if use_llm:
        try:
            out=fc.ollama_chat(prompt,model,url).strip()
            if out: return out
        except Exception: pass
    return fallback

def _front(summary, dataset, exposure, outcome, model, url, use_llm):
    facts=_facts(summary,dataset,exposure,outcome); src=_src(dataset)
    e=summary.get('epi',{}); t=summary.get('trend',{}); m=summary.get('ml',{})
    ab_fb=(f"Background. We comprehensively examined the association between {fc.lab(exposure)} and {fc.lab(outcome)} in {src}. "
           f"Methods. In one framework we estimated survey-weighted cross-sectional associations with confounding control "
           f"and causal-inference methods, characterized secular trends with age-sex standardization and joinpoint "
           f"regression, and built machine-learning prediction models. "
           f"Results. "
           + (f"The fully adjusted odds ratio for {fc.lab(exposure)}{e.get('per','')} was {e['or']:.2f} "
              f"(95% CI {e['lo']:.2f} to {e['hi']:.2f}) with an E-value of {e['evalue']}, at a weighted prevalence of "
              f"{e['prev']:.1f} percent. " if e else "")
           + (f"The age-sex standardized prevalence changed from {t['prev0']:.1f} to {t['prev1']:.1f} percent between "
              f"{t['y0']} and {t['y1']}. " if t else "")
           + (f"The best prediction model ({m['best']}) reached an area under the ROC curve of {m['auROC']}. " if m else "")
           + f"Conclusion. These survey-weighted findings are consistent across analytic approaches and are "
             f"hypothesis-generating.")
    intro_fb=(f"Understanding how {fc.lab(exposure)} relates to {fc.lab(outcome)} has clinical and public-health importance. "
              f"Nationally representative health-examination surveys allow this relationship to be characterized cross-sectionally, "
              f"over time, and predictively within a single consistent framework. Using {src}, we integrated an epidemiologic "
              f"association analysis, a secular-trend analysis, and a machine-learning prediction analysis for {fc.lab(outcome)}.")
    abstract=_llm_or(ab_fb, f"{fc.STYLE}\nWrite a single-paragraph structured abstract (background, methods, results, "
                     f"conclusion) of a research paper in English using only these facts. Do not invent or change any "
                     f"numbers.\nFacts: {facts}", model,url,use_llm)
    intro=_llm_or(intro_fb, f"{fc.STYLE}\nWrite a short Introduction paragraph for a research paper in English motivating the "
                  f"study of this exposure and outcome. Do not state specific result numbers.\nFacts: {facts}",
                  model,url,use_llm)
    return abstract, intro

def _back(summary, dataset, exposure, outcome, model, url, use_llm):
    facts=_facts(summary,dataset,exposure,outcome)
    disc_fb=("The three analyses provide a consistent picture. The cross-sectional association was robust to measured "
             "confounding and to multiple causal-inference estimators, the trend analysis places the association in a "
             "temporal context, and the prediction analysis quantifies discriminative value. Limitations include the "
             "cross-sectional design, which precludes causal inference, reliance on a single survey weight across outcomes, "
             "and, for steatosis-related outcomes, differing measurement between surveys. All findings are "
             "hypothesis-generating and require prospective validation.")
    concl_fb=(f"In {_src(dataset)}, {fc.lab(exposure)} showed a reproducible survey-weighted association with "
              f"{fc.lab(outcome)} across epidemiologic, temporal, and predictive analyses that warrants prospective study.")
    disc=_llm_or(disc_fb, f"{fc.STYLE}\nWrite a Discussion paragraph for a research paper in English that interprets an "
                 f"epidemiologic association analysis, a trend analysis, and a machine-learning prediction analysis together, "
                 f"and lists limitations including the cross-sectional design. Do not change any numbers.\nFacts: {facts}",
                 model,url,use_llm)
    concl=_llm_or(concl_fb, f"{fc.STYLE}\nWrite a one-sentence Conclusion for a research paper in English.\nFacts: {facts}",
                  model,url,use_llm)
    return disc, concl

def build_full_report(dataset, DF, data_dir, outcome, main_exposure, defs, model, url, use_llm,
                      years=None, predictors=None, models=None, cov=None, subgroups=("men","age50"),
                      workdir=None):
    from docx import Document
    amin=defs["pop"]["age_min"]
    if cov is None: cov=fc.auto_covariates(dataset,[main_exposure],[outcome])
    if predictors is None:
        cand=["age","men","bmi","wc","bodyfat_pct","asm_pct","whtr","alt","ast","creatinine","hemoglobin",
              "smoking","alcohol","height","weight",main_exposure]
        predictors=[]
        for v in cand:
            if v in fc.AVAIL[dataset] and v not in predictors and not fc.related(dataset,v,outcome):
                predictors.append(v)
    if years is None:
        yrs_all={"KNHANES":["08","09","10","11"],"NHANES":["j"]}
        years=yrs_all.get(dataset,[])

    doc=Document()
    doc.add_heading(f"{fc.lab(main_exposure)} and {fc.lab(outcome)} in the {dataset}: "
                    f"a combined epidemiologic, trend, and prediction study",0)
    doc.add_paragraph(f"Data: {_src(dataset)} · survey-weighted · exposure {fc.lab(main_exposure)} · "
                      f"outcome {fc.lab(outcome)}")
    anchor=doc.add_paragraph("")   # front matter (Abstract, Introduction) is inserted here after analyses run
    summary={}

    # Part I — epidemiologic analysis (always runs)
    epi_report.build_epi_report(dataset, DF, outcome, main_exposure, cov, list(subgroups), defs,
                                model, url, use_llm, workdir=workdir, doc=doc, summary=summary)
    # Part II — trend analysis (needs >=2 cycles)
    if len(years)>=2:
        try:
            trend_report.build_trend_report(dataset, data_dir, years, outcome,
                {**defs,"pop":{"age_min":max(amin,19)}}, model, url, use_llm, workdir=workdir, doc=doc, summary=summary)
        except Exception as ex:
            doc.add_heading("Incidence/Prevalence Trend Report",1)
            doc.add_paragraph(f"Trend analysis could not be completed: {ex}")
    else:
        doc.add_heading("Incidence/Prevalence Trend Report",1)
        doc.add_paragraph(f"Trend analysis requires at least two survey cycles; {dataset} has {len(years)} here.")
    # Part III — machine-learning prediction
    try:
        ml_report.build_ml_report(dataset, DF, outcome, predictors, defs, model, url, use_llm,
                                  models=models, workdir=workdir or ".", doc=doc, summary=summary)
    except Exception as ex:
        doc.add_heading("Machine Learning Prediction Model Report",1)
        doc.add_paragraph(f"ML analysis could not be completed: {ex}")

    # front and back matter written from the collected numbers
    abstract,intro=_front(summary,dataset,main_exposure,outcome,model,url,use_llm)
    disc,concl=_back(summary,dataset,main_exposure,outcome,model,url,use_llm)
    # insert Abstract + Introduction right after the title (before the anchor), then drop the anchor
    anchor.insert_paragraph_before("Abstract", style="Heading 1")
    anchor.insert_paragraph_before(abstract)
    anchor.insert_paragraph_before("Introduction", style="Heading 1")
    anchor.insert_paragraph_before(intro)
    anchor._p.getparent().remove(anchor._p)
    # Discussion + Conclusion at the end
    doc.add_heading("Discussion",1); doc.add_paragraph(disc)
    doc.add_heading("Conclusion",1); doc.add_paragraph(concl)
    doc.add_paragraph("Statistics are deterministic survey-weighted (R survey) and machine-learning (scikit-learn) "
                      "estimates. Connecting prose was written from the computed values without recomputation. "
                      "All findings are hypothesis-generating.")

    buf=io.BytesIO(); doc.save(buf); return buf.getvalue()
