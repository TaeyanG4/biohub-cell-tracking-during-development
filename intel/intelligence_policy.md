# Biohub Intelligence Operating Policy

Last updated: 2026-09-17 KST
Status: ACTIVE

## Purpose

Information gathering for this competition is a shared responsibility between the user and the AI.
The user may manually collect notebooks, discussions, papers, repositories, model artifacts, or other sources at any time.
The AI must also gather information proactively whenever fresh external evidence could materially improve a research decision.

This is not a passive bookmark archive. The goal is to turn new evidence into testable hypotheses while avoiding leaderboard gossip, stale metric artifacts, duplicated forks, and untrusted code execution.

## AI proactive-gathering rule

The AI should initiate intelligence gathering without waiting for an explicit user request when any of the following is true:

1. A new research branch is about to consume meaningful compute and external work may already solve the same failure mode.
2. Two or more serious experiments fail to improve the trustworthy validation surface and the project appears stalled.
3. Local validation and Kaggle leaderboard behavior disagree materially.
4. A current bottleneck becomes clearer, such as missing detections, ambiguous parent association, gap recovery, division repair, or runtime.
5. A new model family, paper technique, public notebook, discussion claim, or implementation could change the top research priorities.
6. Competition rules, metric behavior, runtime limits, public notebooks, leaderboard frontier, or organizer guidance may have changed.
7. The deadline is approaching and finalist robustness, packaging, runtime, or external-data legality needs re-verification.
8. The user provides new papers, notebooks, code, or observations that should be cross-checked against the existing intelligence base.

Proactive gathering should be targeted. Search by the current structural problem or failure mode, not only by the competition title.

## Source priority

Use the following evidence order by default:

1. Official Kaggle rules, evaluation, data description, timeline, organizer clarifications, and code-competition constraints.
2. Reproducible scored Kaggle notebooks and authenticated submission evidence.
3. High-signal Kaggle discussions containing validation, error-analysis, metric, or organizer information.
4. Public notebooks/forks, after checking whether the change is algorithmic or only cosmetic/config/path changes.
5. Papers, Cell Tracking Challenge methods, related competition writeups, and official model cards.
6. Official or author GitHub repositories and released weights.
7. Generic technique catalogs, blogs, or community commentary only as idea sources, not primary evidence.

## Required processing of new information

Do not merely collect links. For each material source:

- record bibliographic/source identity and date checked;
- classify the claim as FACT, REPORTED, INFERRED, or SPECULATIVE;
- identify which Biohub bottleneck it addresses;
- compare it against B0 and the active research baseline;
- note leakage, metric-hack, rule, license, runtime, and reproducibility risks;
- identify the cheapest credible falsification test;
- assign TEST, WATCH, REFERENCE, or REJECT;
- create an idea card only when the source yields a concrete testable hypothesis.

Useful paper or notebook ideas must eventually connect to an experiment hypothesis. A large undigested source collection is not progress.

## Current baseline context

- B0 / immutable score anchor: exact Reyhan public-0.947 source.
- Active structural research: R3 normalized-HOCT target-parent ranking, followed by graph-level conservative edge replacement.
- Strong external-paper directions identified so far include temporal-history association, learned/sparse-supervised 3D motion flow, uncertainty-triggered local repair, appearance-assisted association, and local tissue-flow features.

Always re-read `../HANDOFF.md` before allowing older intelligence notes to override current experiment status.

## Storage contract

- `../external/papers/raw/` - immutable paper PDFs.
- `../external/papers/manifest.csv` - paper provenance and hashes.
- `papers.csv` - paper relevance / experiment-value registry.
- `paper_notes/` - extracted analysis and transfer notes.
- `idea_cards/` - testable hypotheses promoted from evidence.
- `sources/` - source snapshots or structured source notes where useful.
- `daily_delta.md` - only meaningful new intelligence or priority changes.
- `source_registry.csv` - cross-source registry for future manual and AI-collected material.
- `state.json` - lightweight intelligence state and last-check bookkeeping.

## Paper policy

The AI should collect papers proactively when the active bottleneck is not well covered by the current collection. Prefer a few high-transfer papers over broad indiscriminate literature harvesting.

Current high-value search gaps include:

- learned 3D microscopy motion / optical flow;
- temporal or graph/transformer lineage association;
- sparse/semi-supervised 3D cell detection;
- missing-detection and tracklet-gap recovery;
- multiple-hypothesis tracking for dense microscopy;
- division-specific lineage modeling.

Store raw PDFs separately from analysis notes. Do not modify source PDFs in place.

## Stop conditions

Stop intelligence gathering and return to empirical testing when:

- new sources no longer change the top three hypotheses;
- a cheap experiment can answer the question faster than more searching;
- remaining information is mostly leaderboard gossip, duplicated forks, or parameter micro-tuning;
- enough evidence exists to choose the next falsification test.

## External-action boundary

Proactive research does not authorize external competition actions. The AI may search, inspect, compare, download public research material, and prepare local experiments, but Kaggle submissions, publishing, team changes, or other externally visible actions still require explicit user intent.

