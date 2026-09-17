# Analysis code for the supplementary sections

These scripts produce the results reported in Supplementary Appendix S10 to S14 and
Supplementary Figures S1 to S7. They read the two association manifests released with the
corpus (`suppl/_manifest_association_KNHANES.csv`, `_manifest_association_NHANES.csv`) and,
where noted, query PubMed through NCBI E-utilities.

## Corpus structure (Supplementary Appendix S11 to S13, Figures S1 to S4)

| Script | Produces |
|---|---|
| `make_suppl_figs_tables.py` | Figures S1 to S4; `S11_file_inventory.csv`, `S12_outcome_summary.csv` |
| `figstyle.py` | The house style every supplementary figure is drawn in: palette, rc parameters, panel frame, panel letters, label fitting |
| `rebuild_suppl_data_ml.py` | Rebuilds the merged Supplementary Data S5 and S6 bundles from the individual machine-learning reports |

## Cross-survey replication (main text)

| Script | Produces |
|---|---|
| `discovery_replication.py` | Discovery-replication rates in both directions and the cluster-bootstrap confidence interval for the cross-survey correlation |

## Fasting subsample weight (Methods, Results)

| Script | Produces |
|---|---|
| `fasting_weight_sensitivity.py` | Prevalence of every fasting-dependent NHANES outcome under `WTMEC2YR` and `WTSAF2YR` |
| `fasting_weight_associations.py` | Refits the associations for those outcomes under both weights |

Both need the raw NHANES files, which are not redistributed here; download them from NCHS.

## Comparison with the published literature (Supplementary Appendix S10, S14, Figures S5 to S7)

Run in this order. The first two steps query PubMed and take roughly an hour in total.

| Step | Script | Produces |
|---|---|---|
| 1 | `literature_studied_pairs.py` | Retrieves every NHANES/KNHANES-indexed title 2003-2026 by publication year, parses the exposure-outcome pair from the title, maps it onto the platform vocabulary. Writes `literature_titles.csv`, `literature_studied.csv`, `literature_studied_pairs.csv`, `literature_missing_pairs.csv` |
| 2 | `harvest_abstracts.py` | Abstracts for the association papers, and for the prediction papers of each of the 27 clinical outcomes |
| 3 | `extract_and_compare.py` | Direction of association, reported effect sizes, published discrimination values. Writes `S14_direction_concordance.csv`, `S15_ml_benchmark.csv`, Figure S7 |
| 4 | `rebuild_figs_s5_s6.py` | Figures S5 and S6 |
| 5 | `make_fig_s7.py` | Figure S7, standalone: it reads only `S15_ml_benchmark.csv` and `_ml_regen_summary.csv`, so the figure can be redrawn without repeating the retrieval |

`pubmed_terms.py` holds the query fragment for every variable. **Term choice drives every count
in S10 and S14, so it is meant to be read and edited by a domain expert.**

### What is not committed

`literature_titles.csv`, `literature_studied.csv`, `abstracts_association.csv` and
`abstracts_prediction.csv` are caches of PubMed records, together about 18 MB. They are not
redistributed here because NCBI asks that E-utilities output not be bulk-mirrored; every script
regenerates its own cache on first run and resumes if interrupted.

### What is committed and cannot be regenerated

`extraction_precision_sample.csv` and `direction_precision_sample.csv` are the manually
adjudicated samples behind the two precision estimates reported in the Supplement:

- title-level pair extraction, 24 of 30 correct (80%, 95% CI 66-94)
- the adjudication that established that a published ratio for a continuous biomarker is a
  category contrast and cannot be compared with a per-standard-deviation estimate

Each row carries the source title and, for a rejected row, the reason. These are judgements, not
computations, and they are the evidence for the precision figures.

## Known limitation

None of this validates the definition-audit agents themselves. The comparison measures agreement
between the released corpus and the published literature; it says nothing about the sensitivity or
specificity with which an agent detects a definitional error. That remains open and is stated as
such in the manuscript.

## Participant flow and design degrees of freedom (Methods, Supplementary Appendix S15)

| Script | Produces |
|---|---|
| `participant_flow.py` | Walks the same eligibility filter `build_analytic` applies, in the same order, and writes the cascade and the design degrees of freedom to `participant_flow.csv` |
| `make_flow_diagram.py` | Supplementary Figure S8, the two-column participant flow diagram, from that CSV and the two manifests |

`participant_flow.py` needs the raw survey files, which are not redistributed here;
`participant_flow.csv` is committed so the diagram and the reported cascade can be
checked without them.

The cascade this recovers corrected a statement in the manuscript. The figures 110,239
and 47,558 are the counts aged 20 or over, not the counts with a usable complex-survey
design, which are 105,756 and 44,249. Below that there is no single analytic sample
size at all, because every model is fitted on complete cases for its own variables.

## Figure style

`figstyle.py` holds the whole look in one place, so a change of taste is a change in one
file. It carries the palette of the main figures unchanged, plus two colours that an
accessibility check forced: the salmon and the neutral grey of the main figures are 8.8
apart in OKLab and 4.9 apart under deuteranopia, which is not a distinguishable pair, and
they had been placed together in two figures. `PINK_DK` and `GREY_DK` separate those
cases, and in Figure S2 the eligible-exposure bar became an outline rather than a second
fill for the same reason.

Figures carry no title inside the image, because the Supplement prints the caption
directly beneath each one; panels are identified with a bold lower-case letter, as a
journal does. Set `TITLES = True` in `figstyle.py` to get titles back for a standalone
copy of a figure.
