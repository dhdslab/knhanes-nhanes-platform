# The agent layer

Every agent runs on one self-hosted model, **Llama 3.3 70B** (`llama3.3:70b`), served
through Ollama. This is the model used in the manuscript. Each agent does a task that needs
judgement but no arithmetic. The rule is enforced by construction: an agent receives numbers
already computed by the deterministic core and returns a *verdict* or *prose*, never a new
number.

If Ollama is unavailable, each agent falls back to a deterministic rule or template. The
`source` field of each result records `"llm"` or `"fallback"`, which is why the repository
runs and is tested without a GPU.

| Agent (`agents.py`) | Input (already computed) | Output | Fallback |
|---|---|---|---|
| `DefinitionAuditorAgent` | Codebook label, definition, our and the official prevalence, medication variable | Verdict and pitfall class | Prevalence divergence (> 2 percentage points) and exam-day-drug rules |
| `GuidelineVerifierAgent` | Outcome and definition | Guideline name and compliance flag | Per-outcome guideline registry |
| `HarmonizationAgent` | Concept and candidate variables per survey | Reconciliation note | Availability check |
| `BatchQCAgent` | The results matrix | QC summary | **Detection is always deterministic** |
| `ReportWriterAgent` | Frozen facts (effect sizes, CIs, q) | Abstract, methods, results, discussion | Templated prose |

## Error classes the auditor targets

1. Missing data coded as absence of disease
2. Examination-day medication proxy used as current treatment
3. Population-inappropriate thresholds
4. Lifetime self-report used as current status
5. Incomplete outcome definitions

The manuscript reports nine error classes found and corrected during development (see
[OPERATIONAL_DEFINITIONS.md](OPERATIONAL_DEFINITIONS.md)). The agents were developed within
the pipeline and have not been evaluated on an independent, blinded benchmark.

## Batch QC detections (deterministic)

- a non-finite estimate or confidence bound
- an odds ratio outside [1/50, 50], which suggests skew or leakage
- an outcome below the event floor (`MIN_EVENTS = 50`)

The agent then writes the human-readable QC summary. The numbers it reports are the ones it
was given.

## Report prose and the numeric guard

For the corpus, the report writer is called through `factory_core.llm_prose`. The call is a
fixed style preamble, a task instruction and the fact string emitted by the deterministic
core. Generated text is accepted only if every number in it is an exact token of the fact
string; otherwise the template is written. `suppl_generator.py` enables this path with
`SUPPL_USE_LLM=1`. The released corpus in `suppl/` was generated without it, so its prose
is the deterministic template text.

## Decision ledger

`Orchestrator` holds the agents on one Ollama client and an append-only `DecisionLog`.
Every agent action and every human ratification is appended and saved to
`decision_log.json`, so a run can be audited step by step.

## Try it

```bash
python pipeline.py --demo          # uses Ollama if present, otherwise the fallback
python pipeline.py --demo --no-llm # force the deterministic path
```
