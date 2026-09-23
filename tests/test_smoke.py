# -*- coding: utf-8 -*-
"""Smoke tests for the agent layer and the end-to-end demo.

These run without the raw data, without R, and without a GPU: the agents are forced
onto their deterministic fallback path (enabled=False), so no network is touched.
"""
import json
import os
import sys

# make the package importable when pytest is run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import (  # noqa: E402
    DefinitionAuditorAgent, GuidelineVerifierAgent, BatchQCAgent, Orchestrator, OllamaClient,
)


def test_ollama_available_is_safe_when_offline():
    # Point at a port nothing is listening on; must return False, never raise.
    assert OllamaClient(url="http://127.0.0.1:1").available() is False


def test_auditor_flags_exam_day_drug_proxy():
    a = DefinitionAuditorAgent(enabled=False)
    f = a.audit({"name": "htn", "drug_var": "HE_HPdr",
                 "our_prevalence": 15.4, "reference_prevalence": 25.2, "reference_name": "HE_HP"})
    assert f.verdict == "flag"
    assert f.issue_class == "exam-day-drug-proxy"
    assert f.source == "fallback"


def test_auditor_flags_prevalence_divergence():
    a = DefinitionAuditorAgent(enabled=False)
    f = a.audit({"name": "obesity", "drug_var": "",
                 "our_prevalence": 5.6, "reference_prevalence": 34.6, "reference_name": "KSSO"})
    assert f.verdict == "flag"
    assert f.issue_class == "prevalence-divergence"


def test_auditor_passes_consistent_definition():
    a = DefinitionAuditorAgent(enabled=False)
    f = a.audit({"name": "dm", "drug_var": "DE1_32",
                 "our_prevalence": 13.3, "reference_prevalence": 13.5, "reference_name": "HE_DM_HbA1c"})
    assert f.verdict == "ok"


def test_guideline_verifier_known_outcome():
    g = GuidelineVerifierAgent(enabled=False).verify("ckd", "eGFR<60 or ACR>=30")
    assert g.compliant is True
    assert "KDIGO" in g.guideline


def test_batch_qc_flags_extreme_and_floor():
    qc = BatchQCAgent(enabled=False).scan([
        {"exposure": "x_homa_ir", "outcome": "central_obesity", "measure": "OR",
         "est": 1013.0, "ci_low": 400.0, "ci_high": 2560.0, "events": 900},
        {"exposure": "x_uric", "outcome": "ckd", "measure": "OR",
         "est": 1.4, "ci_low": 1.3, "ci_high": 1.5, "events": 40},
        {"exposure": "x_bmi", "outcome": "mets", "measure": "OR",
         "est": 3.1, "ci_low": 2.8, "ci_high": 3.4, "events": 1800},
    ])
    kinds = {i.kind for i in qc.issues}
    assert "extreme-odds-ratio" in kinds
    assert "below-event-floor" in kinds
    assert qc.n_pairs == 3


def test_orchestrator_offline_banner():
    orch = Orchestrator(use_llm=False)
    assert orch.online is False
    assert "fallback" in orch.banner()


def test_demo_end_to_end(tmp_path):
    import pipeline
    rc = pipeline.main(["--demo", "--no-llm", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "demo_report.docx").exists()
    log = json.loads((tmp_path / "decision_log.json").read_text(encoding="utf-8"))
    assert any(e["agent"] == "definition-auditor" for e in log)
    findings = json.loads((tmp_path / "audit_findings.json").read_text(encoding="utf-8"))
    assert sum(1 for f in findings if f["verdict"] == "flag") == 2
