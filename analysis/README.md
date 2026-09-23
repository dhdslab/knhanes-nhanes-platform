# Analysis code for the Supplementary Appendix

These scripts produce the results reported in Supplementary Appendix S10 to S16 and the data
behind the Supplementary Figures. They read the released corpus in `../suppl/`, mainly the two
association manifests, and `../_ml_regen_summary.csv`. Where noted, they query PubMed
through NCBI E-utilities or need the raw survey files in `../data/`.

Run them from anywhere; paths resolve from the repository. Figures and merged bundles are
written to `analysis/out/`, which git ignores. The final artwork in the submission was
restyled from these drafts. The draft file names follow an earlier numbering, so the table
below gives each draft's number in the final Supplement.

| Draft file | Script | Final Supplementary Figure |
|---|---|---|
| (data only) | `publication_counts_2003_2026.py` | S1, annual publications (and Table S10.3) |
| `SupplFigureS1.png` | `make_suppl_figs_tables.py` | S2, evidence across all association models |
| `SupplFigureS2.png` | `make_suppl_figs_tables.py` | S3, analysis yield per outcome |
| `SupplFigureS3.png` | `make_suppl_figs_tables.py` | S4, contributing participant counts |
| `SupplFigureS4.png` | `make_suppl_figs_tables.py` | S5, cross-survey concordance by measurement family |
| `SupplFigureS5.png` | `rebuild_figs_s5_s6.py` | S6, literature-direction concordance |
| `SupplFigureS6.png` | `rebuild_figs_s5_s6.py` | S7, pair-level literature evidence |
| `SupplFigureS7.png` | `make_fig_s7.py` | S8, prediction models against published discrimination |
| `SupplFigureS8.png` | `make_flow_diagram.py` | S9, participant flow |

Supplementary Figure S10, the NHANES exposure-by-outcome matrix, is drawn from
`../suppl/_manifest_association_NHANES.csv` in the same way as main-text Figure 4, which
shows the KNHANES matrix.

## Publication counts (Supplementary Figure S1, Table S10.3)

| Script | Produces |
|---|---|
| `publication_counts_2003_2026.py` | Annual PubMed counts for each survey, 2003 to 2026, from one query date. Writes `publication_counts_2003_2026.csv` and the exact queries to `publication_counts_2003_2026.queries.json` (both committed) |

The full name of the US survey is contained in that of the Korean survey. The NHANES query
therefore excludes records matched only through the Korean name; the script also records
the uncorrected count, so the size of that double count is on file. 2026 is a partial
year.

## Corpus structure (Supplementary Appendix S11 to S13)

| Script | Produces |
|---|---|
| `make_suppl_figs_tables.py` | Draft figures S1 to S4 (final S2 to S5); `S11_file_inventory.csv` (reads the merged bundles from `out/Suppl/` when present); `S12_outcome_summary.csv` |
| `figstyle.py` | The house style of the draft figures: palette, rc parameters, panel frame, panel letters, label fitting |
| `rebuild_suppl_data_ml.py` | Merges the individual prediction reports into the Supplementary Data S5 and S6 bundles (written to `out/Suppl/`) |

## Cross-survey replication (main text)

| Script | Produces |
|---|---|
| `discovery_replication.py` | Discovery-replication rates in both directions and the cluster-bootstrap confidence interval for the cross-survey correlation |

## Fasting subsample weight (Methods, Results)

| Script | Produces |
|---|---|
| `fasting_weight_sensitivity.py` | Prevalence of every fasting-dependent NHANES outcome under `WTMEC2YR` and `WTSAF2YR` |
| `fasting_weight_associations.py` | Refits the associations for those outcomes under both weights |

Both need the raw NHANES files in `../data/NHANES/`. Their outputs,
`fasting_weight_sensitivity.csv` and `fasting_weight_associations.csv`, are committed.

## Comparison with the published literature (Supplementary Appendix S10, S14)

Run in this order. The first two steps query PubMed and take roughly an hour in total.

| Step | Script | Produces |
|---|---|---|
| 1 | `literature_studied_pairs.py` | Retrieves every NHANES/KNHANES-indexed title 2003–2026 by publication year, parses the exposure–outcome pair from the title and maps it onto the platform vocabulary. Writes `literature_titles.csv`, `literature_studied.csv`, `literature_studied_pairs.csv`, `literature_missing_pairs.csv` |
| 2 | `harvest_abstracts.py` | Abstracts for the association papers, and for the prediction papers of each of the 27 clinical outcomes |
| 3 | `extract_and_compare.py` | Direction of association, reported effect sizes, published discrimination values. Writes `S14_direction_concordance.csv`, `S15_ml_benchmark.csv` |
| 4 | `rebuild_figs_s5_s6.py` | Draft figures S5 and S6 (final S6 and S7) |
| 5 | `make_fig_s7.py` | Draft figure S7 (final S8), standalone: it reads only `S15_ml_benchmark.csv` and `../_ml_regen_summary.csv`, so the figure can be redrawn without repeating the retrieval |

`pubmed_terms.py` holds the query fragment for every variable. **Term choice drives every count
in S10 and S14, so it is meant to be read and edited by a domain expert.**

### What is not committed

`literature_titles.csv`, `literature_studied.csv`, `abstracts_association.csv` and
`abstracts_prediction.csv` are caches of PubMed records, together about 18 MB. They are not
redistributed because NCBI asks that E-utilities output not be bulk-mirrored. Every script
regenerates its own cache on first run and resumes if interrupted.

### What is committed and cannot be regenerated

`extraction_precision_sample.csv` and `direction_precision_sample.csv` are the manually
adjudicated samples behind the two precision estimates reported in the Supplement:

- title-level pair extraction: 24 of 30 correct (80%, 95% CI 66–94)
- the adjudication establishing that a published ratio for a continuous biomarker is a
  category contrast, which cannot be compared with a per-standard-deviation estimate

Each row carries the source title and, for a rejected row, the reason. These are judgements,
not computations, and they are the evidence for the precision figures.

## Known limitation

None of this validates the definition-audit agents themselves. The comparison measures
agreement between the released corpus and the published literature. It says nothing about
how sensitively or specifically an agent detects a definitional error. That remains open,
and the manuscript says so.

## Participant flow and design degrees of freedom (Supplementary Appendix S15)

| Script | Produces |
|---|---|
| `participant_flow.py` | Walks the eligibility filter of `build_analytic`, in the same order, and writes the cascade and the design degrees of freedom to `participant_flow.csv` |
| `make_flow_diagram.py` | Draft figure S8 (final S9), the two-column participant flow diagram, from that CSV and the two manifests |

`participant_flow.py` needs the raw survey files. `participant_flow.csv` is committed, so the
diagram and the reported cascade can be checked without them.

The cascade corrected a statement in the manuscript. The figures 110,239 and 47,558 are the
counts aged 20 or over. The counts with a usable complex-survey design are 105,756 and
44,249. Below that there is no single analytic sample size, because every model is fitted on
the participants with its own variables. The association screen also uses a fixed random
sample of 50,000 pooled records per survey (see the main README), so an association's
contributing N is at most 37,636 (KNHANES) and 26,892 (NHANES).

## Figure style

`figstyle.py` holds the whole look of the draft figures in one place, so a change of taste
is a change in one file. It keeps the palette of the main figures, plus two colours added
after an accessibility check. The salmon and the neutral grey of the main figures are 8.8
apart in OKLab and 4.9 apart under deuteranopia, which is not a distinguishable pair, and
they had been placed together in two figures. `PINK_DK` and `GREY_DK` separate those cases.
For the same reason, the eligible-exposure bar in the yield figure became an outline rather
than a second fill.

Figures carry no title inside the image, because the Supplement prints the caption directly
beneath each one. Panels are identified with a bold lower-case letter, as a journal does.
Set `TITLES = True` in `figstyle.py` to get titles back for a standalone copy of a figure.
