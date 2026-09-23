# -*- coding: utf-8 -*-
"""
pipeline.py — end-to-end orchestrator for the autonomous AI co-scientist
========================================================================

One command takes raw survey files to a finished Word report:

    raw KNHANES/.sas7bdat, NHANES/.XPT
        -> harmonized definitions            (factory_core)
        -> agent audit of every definition   (agents: guideline-verifier, def-auditor)
        -> deterministic complex-survey stats (factory_core -> R survey)
        -> batch quality control             (agents: batch-qc, numeric detection)
        -> report prose written by Llama-3.3 (agents: report-writer)
        -> Word (.docx) + machine-readable results + a logged decision ledger

The language model audits, coordinates, and writes; every number is computed by the
deterministic core. If Ollama is offline the agents fall back to deterministic rules
and templates, so the whole pipeline still runs.

Usage
-----
    # No data, no R, no GPU required. Shows the agents catching planted pitfalls and
    # writing a Word report. Good for a first run and for CI.
    python pipeline.py --demo

    # Real run (needs the private data, R with the survey package, and ideally Ollama):
    python pipeline.py --survey KNHANES --data-dir data/KNHANES \
        --cycles 08 09 10 11 --outcomes dm htn mets --out out/

    # Force the deterministic path even if Ollama is up:
    python pipeline.py --demo --no-llm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from agents import Orchestrator, AuditFinding


# ─────────────────────────────────────────────────────────────────────────────
# Demo: agents-only, no data / no R / no GPU
# ─────────────────────────────────────────────────────────────────────────────
def _demo_prose(orch: Orchestrator, facts: str, sections: List[str]) -> dict:
    """Write each section with the report-writer's Llama client, or template it."""
    out = {}
    if orch.online:
        try:
            for nm in sections:
                out[nm] = orch.writer.client.chat(
                    orch.writer.system,
                    f"Write the {nm} of a short research report using only these facts. "
                    f"Do not change any number.\nFacts: {facts}").strip()
            out["_source"] = "llm"
            return out
        except Exception:
            pass
    tmpl = {
        "Methods": ("We analysed survey-weighted, cross-sectional data with the R survey package, "
                    "reporting odds ratios from survey logistic regression and controlling the false "
                    "discovery rate. " + facts),
        "Results": "The estimated associations, with confidence intervals and q-values, are listed in the table. " + facts,
        "Discussion": ("These associations are hypothesis-generating and require prospective confirmation. "
                       "The audit layer corrected the definitional pitfalls before any report was released."),
    }
    out = {nm: tmpl.get(nm, facts) for nm in sections}
    out["_source"] = "fallback"
    return out


def run_demo(args) -> int:
    from docx import Document

    orch = Orchestrator(model=args.model, url=args.url, use_llm=not args.no_llm)
    print("[pipeline] " + orch.banner())
    os.makedirs(args.out, exist_ok=True)

    # 1) Guideline verification of a few harmonized outcomes -------------------
    definitions = {
        "dm": "FPG >= 126 mg/dL, or HbA1c >= 6.5%, or current glucose-lowering medication",
        "htn": "SBP >= 140, or DBP >= 90, or current antihypertensive treatment",
        "mets": "harmonized NCEP ATP III, 3 of 5 components, population-specific waist thresholds",
        "ckd": "eGFR < 60 (CKD-EPI 2021) or urine albumin-to-creatinine ratio >= 30 mg/g",
        "masld": "hepatic steatosis and >= 1 cardiometabolic risk factor, alcohol below the MetALD floor",
    }
    guideline_checks = []
    for oc, d in definitions.items():
        gc = orch.guideline.verify(oc, d)
        orch.log.log(orch.guideline.role, f"verify {oc}", gc.__dict__)
        guideline_checks.append(gc.__dict__)
        print(f"  guideline  {oc:6s} -> {gc.guideline}  (compliant={gc.compliant}, {gc.source})")

    # 2) Definition audit, including two PLANTED pitfalls the agent should catch
    audit_items = [
        {"name": "htn", "label": "Hypertension", "drug_var": "HE_HPdr",
         "definition": "SBP/DBP or the examination-day antihypertensive item",
         "our_prevalence": 15.4, "reference_prevalence": 25.2, "reference_name": "HE_HP"},
        {"name": "obesity", "label": "Obesity", "drug_var": "",
         "definition": "BMI >= 30 (WHO) applied to Koreans",
         "our_prevalence": 5.6, "reference_prevalence": 34.6, "reference_name": "KSSO national"},
        {"name": "dm", "label": "Diabetes mellitus", "drug_var": "DE1_32",
         "definition": "FPG/HbA1c or current glucose-lowering medication",
         "our_prevalence": 13.3, "reference_prevalence": 13.5, "reference_name": "HE_DM_HbA1c"},
    ]
    audit_findings: List[AuditFinding] = []
    for item in audit_items:
        f = orch.auditor.audit(item)
        orch.log.log(orch.auditor.role, f"audit {f.variable}", f.as_dict())
        audit_findings.append(f)
        tag = "FLAG" if f.verdict == "flag" else "ok  "
        print(f"  audit      {f.variable:8s} {tag} {f.issue_class or '-':22s} ({f.source})")

    # 3) A small synthetic results matrix with one deliberately extreme OR ------
    results = [
        {"exposure": "x_bmi", "outcome": "mets", "measure": "OR", "est": 3.10,
         "ci_low": 2.80, "ci_high": 3.43, "p": 1e-40, "q": 1e-38, "events": 1800},
        {"exposure": "x_alt", "outcome": "mets", "measure": "OR", "est": 1.88,
         "ci_low": 1.77, "ci_high": 2.01, "p": 2e-70, "q": 2e-68, "events": 1800},
        {"exposure": "x_homa_ir", "outcome": "central_obesity", "measure": "OR", "est": 1013.0,
         "ci_low": 400.0, "ci_high": 2560.0, "p": 1e-9, "q": 1e-8, "events": 900},
        {"exposure": "x_uric", "outcome": "ckd", "measure": "OR", "est": 1.43,
         "ci_low": 1.35, "ci_high": 1.52, "p": 5e-22, "q": 4e-21, "events": 40},  # below floor
    ]
    qc = orch.qc.scan(results)
    orch.log.log(orch.qc.role, "scan corpus", qc.as_dict())
    print(f"  batch-qc   {qc.n_pairs} pairs, {qc.n_flagged} flagged ({qc.source})")
    print(f"             {qc.summary}")

    # 4) Report-writer prose + a Word report -----------------------------------
    facts = ("Survey-weighted cross-sectional analysis. Four exposure-outcome pairs were estimated; "
             "the strongest was body mass index with metabolic syndrome (odds ratio 3.10, 95% CI 2.80 to 3.43). "
             "The audit layer flagged an examination-day medication proxy and an ethnicity-inappropriate "
             "obesity threshold before release.")
    sections = _demo_prose(orch, facts, ["Methods", "Results", "Discussion"])
    orch.log.log(orch.writer.role, "write prose", {"source": sections.get("_source")})

    doc = Document()
    doc.add_heading("Demonstration report: audited AI co-scientist (synthetic data)", 0)
    doc.add_paragraph(f"Prose source: {sections.get('_source')} | {orch.banner()}")
    for nm in ["Methods", "Results", "Discussion"]:
        doc.add_heading(nm, 1)
        doc.add_paragraph(sections.get(nm, ""))
    doc.add_heading("Quality control", 1)
    doc.add_paragraph(qc.summary)
    doc.add_heading("Table. Estimated associations", 1)
    cols = ["Exposure", "Outcome", "Measure", "Estimate (95% CI)", "FDR q"]
    tb = doc.add_table(rows=1, cols=len(cols))
    try:
        tb.style = "Light Grid Accent 1"
    except Exception:
        pass
    for j, c in enumerate(cols):
        tb.rows[0].cells[j].text = c
    for r in results:
        cc = tb.add_row().cells
        cc[0].text = r["exposure"][2:]
        cc[1].text = r["outcome"]
        cc[2].text = r["measure"]
        cc[3].text = f"{r['est']:.2f} ({r['ci_low']:.2f}-{r['ci_high']:.2f})"
        cc[4].text = f"{r['q']:.2g}"
    doc.add_paragraph("All statistics are deterministic; text was written from the computed values "
                      "without recomputation. Synthetic data; illustrative only.")
    report_path = os.path.join(args.out, "demo_report.docx")
    doc.save(report_path)

    # 5) Persist the audit trail ------------------------------------------------
    with open(os.path.join(args.out, "guideline_checks.json"), "w", encoding="utf-8") as fh:
        json.dump(guideline_checks, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "audit_findings.json"), "w", encoding="utf-8") as fh:
        json.dump([f.as_dict() for f in audit_findings], fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "qc_report.json"), "w", encoding="utf-8") as fh:
        json.dump(qc.as_dict(), fh, ensure_ascii=False, indent=2)
    orch.log.save(os.path.join(args.out, "decision_log.json"))

    n_flag = sum(1 for f in audit_findings if f.verdict == "flag")
    print(f"[pipeline] wrote {report_path}")
    print(f"[pipeline] audit flagged {n_flag}/{len(audit_findings)} definitions; "
          f"QC flagged {qc.n_flagged}/{qc.n_pairs} pairs; ledger + JSON written to {args.out}/")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Real run: raw data -> R core -> Word
# ─────────────────────────────────────────────────────────────────────────────
def run_real(args) -> int:
    import factory_core as fc

    orch = Orchestrator(model=args.model, url=args.url, use_llm=not args.no_llm)
    print("[pipeline] " + orch.banner())
    os.makedirs(args.out, exist_ok=True)

    cycles = args.cycles or []
    print(f"[pipeline] loading {args.survey} from {args.data_dir} (cycles={cycles or 'all'})")
    d = fc.load_raw(args.survey, args.data_dir, cycles)

    outcomes = args.outcomes or [o for o in fc.outcome_candidates(args.survey)]
    exposures = args.exposures or [e for e in fc.exposure_candidates(args.survey)]
    outcomes = [o for o in outcomes if o in fc.AVAIL.get(args.survey, [])]
    print(f"[pipeline] {len(exposures)} exposures x {len(outcomes)} outcomes")

    # Agent audit of the definitions (guideline verification is always available).
    for o in outcomes:
        gc = orch.guideline.verify(o, fc.lab(o))
        orch.log.log(orch.guideline.role, f"verify {o}", gc.__dict__)

    # Deterministic core computes the whole matrix.
    def _progress(i, n, o):
        print(f"    [{i + 1:>3}/{n}] {fc.lab(o)}")
    RES, info, t1 = fc.run_matrix(d, exposures, outcomes, args.survey, args.age_min,
                                  workdir=args.out, progress=_progress)
    print(f"[pipeline] {len(RES)} associations; "
          f"{int((RES.q < 0.05).sum()) if len(RES) else 0} significant at q<0.05")

    # Batch QC (numeric detection + agent prose).
    qc = orch.qc.scan(RES.to_dict("records") if len(RES) else [])
    orch.log.log(orch.qc.role, "scan corpus", {"n_pairs": qc.n_pairs, "n_flagged": qc.n_flagged})
    print(f"[pipeline] QC flagged {qc.n_flagged}/{qc.n_pairs} ({qc.source})")

    # Report-writer prose, then the Word manuscript.
    sections = orch.writer.write_full(RES, info, args.survey, exposures, outcomes)
    orch.log.log(orch.writer.role, "write full manuscript", {"source": sections.get("_source")})
    docx = fc.build_full_manuscript_docx(RES, info, sections, args.survey, exposures, outcomes, t1)
    report_path = os.path.join(args.out, f"manuscript_{args.survey}.docx")
    with open(report_path, "wb") as fh:
        fh.write(docx)

    if len(RES):
        RES.to_csv(os.path.join(args.out, f"results_{args.survey}.csv"), index=False, encoding="utf-8-sig")
    info.to_csv(os.path.join(args.out, f"outcome_status_{args.survey}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(args.out, "qc_report.json"), "w", encoding="utf-8") as fh:
        json.dump(qc.as_dict(), fh, ensure_ascii=False, indent=2)
    orch.log.save(os.path.join(args.out, "decision_log.json"))
    print(f"[pipeline] wrote {report_path} and machine-readable results to {args.out}/")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Autonomous AI co-scientist pipeline (KNHANES/NHANES).")
    p.add_argument("--demo", action="store_true",
                   help="run the agents-only demonstration (no data, R, or GPU required)")
    p.add_argument("--survey", choices=["KNHANES", "NHANES"], help="survey for a real run")
    p.add_argument("--data-dir", help="directory of raw .sas7bdat / .XPT files")
    p.add_argument("--cycles", nargs="*", help="cycle identifiers to include (default: all found)")
    p.add_argument("--outcomes", nargs="*", help="outcome keys (default: all harmonized outcomes)")
    p.add_argument("--exposures", nargs="*", help="exposure keys (default: all candidate exposures)")
    p.add_argument("--age-min", type=int, default=20, help="minimum age (default 20)")
    p.add_argument("--out", default="out", help="output directory (default ./out)")
    p.add_argument("--model", default="llama3.3:70b", help="Ollama model tag")
    p.add_argument("--url", default="http://localhost:11434", help="Ollama server URL")
    p.add_argument("--no-llm", action="store_true", help="force the deterministic fallback path")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.demo:
        return run_demo(args)
    if not (args.survey and args.data_dir):
        print("error: a real run needs --survey and --data-dir (or use --demo).", file=sys.stderr)
        return 2
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
