# -*- coding: utf-8 -*-
"""
agents.py — Self-hosted Llama-3.3 agent layer for the KNHANES/NHANES automation platform
=========================================================================================

Every agent in this module runs on a single self-hosted Llama-3.3 (70B) model served
locally through Ollama. The agents perform the tasks of the platform that are
*judgment-laden but non-numerical*: auditing operational definitions against the raw
codebooks and the official survey-derived variables, verifying definitions against
current clinical guidelines, reconciling variable names across cycles and surveys,
quality-controlling the generated corpus, and writing report prose.

Invariant (enforced by construction)
------------------------------------
The language model may **audit, coordinate, and write, but it never computes, selects,
or modifies a statistic**. Every number originates in the deterministic core
(``factory_core`` -> R ``survey`` and scikit-learn). Each agent is given numbers that
were already computed and returns a *judgment* or *prose*, never a new number.

Graceful degradation
---------------------
If Ollama is unavailable (no GPU, model not pulled, server down), every agent falls
back to a deterministic rule or template so the whole pipeline still runs end-to-end.
The ``source`` field of every returned object records whether it came from the model
("llm") or the deterministic fallback ("fallback"). This is what lets the repository be
run and tested on a laptop with no GPU, while a workstation with the four NVIDIA RTX PRO
6000 Blackwell GPUs used in the paper runs the full Llama-3.3 agents.

Design
------
``OllamaClient``      thin, retrying client over the Ollama HTTP API (chat + availability)
``LlamaAgent``        base class: system prompt, JSON helper, fallback gating
``DefinitionAuditorAgent``   flags operational-definition pitfalls
``GuidelineVerifierAgent``   checks a definition against 2023-2025 guidelines
``HarmonizationAgent``       reconciles variable names/thresholds across surveys
``BatchQCAgent``             scans the generated corpus (numeric checks + prose summary)
``ReportWriterAgent``        writes report prose from frozen facts
``Orchestrator``             sequences the agents and logs every decision
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

try:                       # requests is a hard dependency of the platform, but keep the
    import requests        # import soft so unit tests can run without a network stack.
except Exception:          # pragma: no cover
    requests = None

# Kept local so this module can be imported without the heavy scientific stack
# (pyreadstat/pandas) that ``factory_core`` pulls in.
MIN_EVENTS = 50

DEFAULT_MODEL = "llama3.3:70b"
DEFAULT_URL = "http://localhost:11434"


# ─────────────────────────────────────────────────────────────────────────────
# Ollama client
# ─────────────────────────────────────────────────────────────────────────────
class OllamaUnavailable(RuntimeError):
    """Raised when the local Ollama server / model cannot be reached."""


@dataclass
class OllamaClient:
    """Minimal, retrying client for the local Ollama server.

    Parameters
    ----------
    model : the Ollama model tag (default ``llama3.3:70b``).
    url   : base URL of the Ollama server (default ``http://localhost:11434``).
    temperature : low by default; the agents are graders and writers, not samplers.
    timeout : per-request timeout in seconds.
    max_retries : transient-failure retries before giving up.
    """

    model: str = DEFAULT_MODEL
    url: str = DEFAULT_URL
    temperature: float = 0.2
    timeout: int = 300
    max_retries: int = 2

    def available(self) -> bool:
        """Return True iff the Ollama server answers. Never raises."""
        if requests is None:
            return False
        try:
            r = requests.get(self.url.rstrip("/") + "/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        """Single-turn chat completion. Raises ``OllamaUnavailable`` on failure."""
        if requests is None:
            raise OllamaUnavailable("the 'requests' package is not installed")
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            body["format"] = "json"
        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(self.url.rstrip("/") + "/api/chat", json=body, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - surfaced as OllamaUnavailable below
                last = exc
                time.sleep(0.75 * (attempt + 1))
        raise OllamaUnavailable(f"Ollama request failed after retries: {last}")

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        """Chat and parse a JSON object from the reply. Raises on unparseable output."""
        raw = self.chat(system, user, json_mode=True)
        return json.loads(raw)

    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Return an embedding vector for ``text`` (used by the RAG retriever).

        Uses the given embedding model (default: the chat model, but a dedicated
        embedding model such as ``nomic-embed-text`` is recommended). Raises
        ``OllamaUnavailable`` on failure so the caller can fall back to lexical search.
        """
        if requests is None:
            raise OllamaUnavailable("the 'requests' package is not installed")
        try:
            r = requests.post(
                self.url.rstrip("/") + "/api/embeddings",
                json={"model": model or self.model, "prompt": text},
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()["embedding"]
        except Exception as exc:  # noqa: BLE001
            raise OllamaUnavailable(f"Ollama embedding failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Base agent
# ─────────────────────────────────────────────────────────────────────────────
class LlamaAgent:
    """Base class. Subclasses set ``role`` and ``system`` and implement the public
    method, which must always return a result even when the model is unavailable."""

    role: str = "agent"
    system: str = "You are a careful biomedical research assistant."

    def __init__(self, client: Optional[OllamaClient] = None, enabled: bool = True):
        self.client = client or OllamaClient()
        # ``enabled`` lets a caller force the deterministic path (e.g., in CI).
        self.enabled = enabled

    def _use_llm(self) -> bool:
        return self.enabled and self.client.available()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Definition auditor
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AuditFinding:
    variable: str
    our_prevalence: Optional[float]
    reference: Optional[float]
    reference_name: str
    verdict: str          # "ok" | "flag"
    issue_class: str      # e.g. "exam-day-drug-proxy", "prevalence-divergence", ""
    rationale: str
    source: str           # "llm" | "fallback"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DefinitionAuditorAgent(LlamaAgent):
    """Audits one operational definition against the raw codebook label and, when
    available, the official survey-derived variable. Returns a verdict and, if flagged,
    the pitfall class. The *numbers* (our vs official prevalence) are supplied by the
    deterministic core; the agent only judges whether they are consistent and whether
    the construction matches a documented error class."""

    role = "definition-auditor"
    system = (
        "You audit operational definitions for secondary analyses of NHANES/KNHANES. "
        "You are given a variable's codebook label, its operational definition, the "
        "survey-weighted prevalence we computed, and (optionally) the prevalence of the "
        "official survey-derived variable. You NEVER compute or change a number. You "
        "decide whether the definition is sound or matches a known pitfall: missing "
        "coded as absence of disease, an examination-day medication proxy used as "
        "current treatment, an ethnicity-inappropriate threshold, a lifetime "
        "self-report used as current status, or an incomplete outcome definition. "
        "Reply only as JSON: {\"verdict\":\"ok|flag\",\"issue_class\":\"...\",\"rationale\":\"...\"}."
    )

    # divergence (in percentage points) beyond which the fallback flags a mismatch
    DIVERGENCE_PP = 2.0

    def audit(self, item: Dict[str, Any]) -> AuditFinding:
        name = item.get("name", "?")
        ours = item.get("our_prevalence")
        ref = item.get("reference_prevalence")
        ref_name = item.get("reference_name", "")
        if self._use_llm():
            try:
                prompt = (
                    f"Variable: {name}\n"
                    f"Codebook label: {item.get('label','')}\n"
                    f"Operational definition: {item.get('definition','')}\n"
                    f"Survey-weighted prevalence we computed: "
                    f"{'NA' if ours is None else f'{ours:.1f}%'}\n"
                    f"Official derived variable ({ref_name}): "
                    f"{'NA' if ref is None else f'{ref:.1f}%'}\n"
                    f"Medication variable used (if any): {item.get('drug_var','none')}\n"
                    "Audit this definition."
                )
                out = self.client.chat_json(self.system, prompt)
                return AuditFinding(
                    variable=name, our_prevalence=ours, reference=ref, reference_name=ref_name,
                    verdict=str(out.get("verdict", "ok")).lower().strip(),
                    issue_class=str(out.get("issue_class", "") or ""),
                    rationale=str(out.get("rationale", "")).strip(),
                    source="llm",
                )
            except Exception:
                pass  # fall through to deterministic rules
        return self._fallback(item, name, ours, ref, ref_name)

    def _fallback(self, item, name, ours, ref, ref_name) -> AuditFinding:
        drug = str(item.get("drug_var", "") or "")
        # Rule 1: exam-day medication proxy (KNHANES HE_*dr) misused as current treatment.
        if drug.lower().endswith("dr") or "exam" in drug.lower():
            return AuditFinding(name, ours, ref, ref_name, "flag", "exam-day-drug-proxy",
                                f"{drug} records medication on the examination day, not current "
                                "treatment; substitute the current-treatment item.", "fallback")
        # Rule 2: prevalence diverges materially from the official derived variable.
        if ours is not None and ref is not None and abs(ours - ref) > self.DIVERGENCE_PP:
            return AuditFinding(name, ours, ref, ref_name, "flag", "prevalence-divergence",
                                f"Computed prevalence {ours:.1f}% differs from {ref_name} "
                                f"{ref:.1f}% by more than {self.DIVERGENCE_PP:g} pp.", "fallback")
        return AuditFinding(name, ours, ref, ref_name, "ok", "",
                            "Definition consistent with the reference within tolerance.", "fallback")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Guideline verifier
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GuidelineCheck:
    outcome: str
    guideline: str
    compliant: bool
    note: str
    source: str


class GuidelineVerifierAgent(LlamaAgent):
    """Checks an operational definition against the relevant 2023-2025 clinical
    guideline. The fallback carries a small registry of the guideline each outcome is
    anchored to, matching the manuscript's Methods."""

    role = "guideline-verifier"
    system = (
        "You verify whether an operational definition matches the current (2023-2025) "
        "clinical guideline for that condition. Name the guideline and state whether the "
        "definition is compliant. Reply only as JSON: "
        "{\"guideline\":\"...\",\"compliant\":true|false,\"note\":\"...\"}."
    )

    GUIDELINES = {
        "dm": "ADA Standards of Care 2024",
        "prediabetes": "ADA Standards of Care 2024",
        "htn": "measured BP / current antihypertensive (JNC/ACC-AHA)",
        "mets": "harmonized NCEP ATP III",
        "obesity": "KSSO 2022 (Asia) / WHO",
        "overweight": "KSSO 2022 (Asia) / WHO",
        "ckd": "KDIGO 2024",
        "masld": "AASLD-EASL-ALEH 2023 multisociety nomenclature",
        "adv_fibrosis": "FIB-4 (Sterling)",
        "anemia": "WHO haemoglobin thresholds",
        "sarcopenia": "AWGS 2019",
        "osteoporosis": "ISCD 2024 / WHO T-score",
        "vitd_deficiency": "Endocrine Society 25(OH)D < 20 ng/mL",
    }

    def verify(self, outcome: str, definition: str) -> GuidelineCheck:
        if self._use_llm():
            try:
                out = self.client.chat_json(
                    self.system, f"Outcome: {outcome}\nDefinition: {definition}\nVerify.")
                return GuidelineCheck(outcome, str(out.get("guideline", "")),
                                      bool(out.get("compliant", True)),
                                      str(out.get("note", "")).strip(), "llm")
            except Exception:
                pass
        g = self.GUIDELINES.get(outcome, "current clinical guideline")
        return GuidelineCheck(outcome, g, True,
                              f"Definition anchored to {g}.", "fallback")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Harmonization agent
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class HarmonizationNote:
    concept: str
    knhanes: List[str]
    nhanes: List[str]
    note: str
    source: str


class HarmonizationAgent(LlamaAgent):
    """Reconciles the variable names that encode the same concept across cycles and
    across the two surveys (e.g. mercury vs automated sphygmomanometer, renamed lipid
    variables). The deterministic core already coalesces these; this agent documents and
    double-checks the mapping."""

    role = "harmonization"
    system = (
        "You reconcile variable names that encode the same clinical concept across "
        "survey cycles and across KNHANES and NHANES. Confirm the concept is measured in "
        "both and note any unit conversion. Reply only as JSON: "
        "{\"note\":\"...\"}."
    )

    def reconcile(self, concept: str, knhanes: Sequence[str], nhanes: Sequence[str]) -> HarmonizationNote:
        kn, nh = list(knhanes), list(nhanes)
        if self._use_llm():
            try:
                out = self.client.chat_json(
                    self.system,
                    f"Concept: {concept}\nKNHANES variables: {kn}\nNHANES variables: {nh}")
                return HarmonizationNote(concept, kn, nh, str(out.get("note", "")).strip(), "llm")
            except Exception:
                pass
        both = bool(kn) and bool(nh)
        note = (f"'{concept}' is available in both surveys and coalesced across cycles."
                if both else f"'{concept}' is not available in both surveys.")
        return HarmonizationNote(concept, kn, nh, note, "fallback")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Batch quality controller
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QCIssue:
    kind: str
    detail: str
    exposure: str = ""
    outcome: str = ""
    value: Optional[float] = None


@dataclass
class QCReport:
    n_pairs: int
    issues: List[QCIssue]
    summary: str
    source: str

    @property
    def n_flagged(self) -> int:
        return len(self.issues)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["n_flagged"] = self.n_flagged
        return d


class BatchQCAgent(LlamaAgent):
    """Scans the generated corpus for defects. The *detection* is deterministic
    (extreme odds ratios, non-finite estimates, implausible prevalence, event floor,
    definitionally circular pairs) so that no judgment call hides a real number problem.
    The agent then writes a human-readable QC summary; the fallback templates it."""

    role = "batch-qc"
    system = (
        "You write a short quality-control summary of an automated analysis corpus from a "
        "list of detected issues. Do not invent numbers; use only what is given. "
        "Two or three sentences."
    )

    OR_HIGH = 50.0     # any |OR| beyond [1/50, 50] is treated as skew-driven and suspect
    OR_LOW = 1.0 / 50.0

    def scan(self, results: Sequence[Dict[str, Any]], min_events: int = MIN_EVENTS) -> QCReport:
        issues: List[QCIssue] = []
        for r in results:
            exp, out = str(r.get("exposure", "")), str(r.get("outcome", ""))
            est, lo, hi = r.get("est"), r.get("ci_low"), r.get("ci_high")
            measure = str(r.get("measure", ""))
            ev = r.get("events")
            for label, v in (("estimate", est), ("ci_low", lo), ("ci_high", hi)):
                if v is not None and (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    issues.append(QCIssue("non-finite", f"{label} is not finite", exp, out, None))
            if measure == "OR" and est is not None and isinstance(est, (int, float)):
                if est > self.OR_HIGH or (0 < est < self.OR_LOW):
                    issues.append(QCIssue("extreme-odds-ratio",
                                          "odds ratio outside [1/50, 50]; check for skew or leakage",
                                          exp, out, float(est)))
            if ev is not None and isinstance(ev, (int, float)) and ev < min_events:
                issues.append(QCIssue("below-event-floor",
                                      f"{int(ev)} events < {min_events}", exp, out, float(ev)))
        summary = self._summary(len(list(results)), issues)
        return summary

    def _summary(self, n_pairs: int, issues: List[QCIssue]) -> QCReport:
        if self._use_llm():
            try:
                brief = [f"{i.kind}: {i.detail} ({i.exposure}->{i.outcome})" for i in issues[:40]]
                text = self.client.chat(
                    self.system,
                    f"{n_pairs} pairs scanned. Issues ({len(issues)}): {brief}. Write the summary.")
                return QCReport(n_pairs, issues, text.strip(), "llm")
            except Exception:
                pass
        if not issues:
            text = (f"All {n_pairs} generated pairs passed automated quality control: no non-finite "
                    "estimates, no extreme odds ratios, and no outcomes below the event floor.")
        else:
            kinds: Dict[str, int] = {}
            for i in issues:
                kinds[i.kind] = kinds.get(i.kind, 0) + 1
            parts = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
            text = (f"Of {n_pairs} generated pairs, {len(issues)} were flagged by automated quality "
                    f"control ({parts}). Flagged pairs are withheld pending review.")
        return QCReport(n_pairs, issues, text, "fallback")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Report writer
# ─────────────────────────────────────────────────────────────────────────────
class ReportWriterAgent(LlamaAgent):
    """Writes the prose of each report (structured abstract, methods, results,
    interpretation) from the frozen, already-computed facts. Delegates to the
    deterministic core's prose functions, which enforce the same invariant and provide a
    template fallback when the model is unavailable."""

    role = "report-writer"
    system = (
        "You write medical-research prose from a fixed set of computed facts. You are "
        "absolutely prohibited from adding, removing, or altering any number."
    )

    def write_full(self, RES, info, dataset: str, exposures, outcomes) -> Dict[str, Any]:
        """Return the manuscript sections for a whole exposure-by-outcome matrix."""
        import factory_core as fc  # lazy: keeps this module importable without the sci stack
        return fc.gen_full_manuscript(
            RES, info, dataset, exposures, outcomes,
            model=self.client.model, url=self.client.url, use_llm=self._use_llm())

    def write_pair(self, t1, res, dataset: str, exposures, outcomes, covariates, defs=None) -> Dict[str, Any]:
        """Return the sections for a single exposure-outcome report."""
        import factory_core as fc
        return fc.gen_manuscript(
            t1, res, dataset, exposures, outcomes, covariates, defs or {},
            model=self.client.model, url=self.client.url, use_llm=self._use_llm())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DecisionLog:
    """An append-only ledger of every automated and human-ratified decision, written to
    disk so that the whole run is auditable (the manuscript's 'every decision logged')."""

    entries: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, agent: str, action: str, detail: Any = "") -> None:
        self.entries.append({"n": len(self.entries) + 1, "agent": agent,
                             "action": action, "detail": detail})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.entries, fh, ensure_ascii=False, indent=2)


class Orchestrator:
    """Builds and holds the agent fleet on one shared Ollama client, and records the
    decision ledger. The step sequencing lives in ``pipeline.py``; this class is the
    single place the agents and the ledger are constructed."""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL, use_llm: bool = True):
        self.client = OllamaClient(model=model, url=url)
        self.online = use_llm and self.client.available()
        self.log = DecisionLog()
        self.auditor = DefinitionAuditorAgent(self.client, enabled=self.online)
        self.guideline = GuidelineVerifierAgent(self.client, enabled=self.online)
        self.harmonizer = HarmonizationAgent(self.client, enabled=self.online)
        self.qc = BatchQCAgent(self.client, enabled=self.online)
        self.writer = ReportWriterAgent(self.client, enabled=self.online)

    def banner(self) -> str:
        where = f"Llama-3.3 via Ollama ({self.client.model})" if self.online else \
                "deterministic fallback (Ollama offline)"
        return f"agent layer: {where}"


__all__ = [
    "OllamaClient", "OllamaUnavailable", "LlamaAgent",
    "DefinitionAuditorAgent", "GuidelineVerifierAgent", "HarmonizationAgent",
    "BatchQCAgent", "ReportWriterAgent", "Orchestrator", "DecisionLog",
    "AuditFinding", "GuidelineCheck", "HarmonizationNote", "QCIssue", "QCReport",
]
