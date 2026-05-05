# Outline

## Title
- Prompt-Only Frontier LLMs for Music Recommendation from Listening History

## Abstract
- State the practical question: can prompt-only LLMs recommend music from exported history.
- Describe the closed-set ranking protocol on Last.fm and Spotify MPD.
- Preview main results with exact MRR/NDCG gains over the strongest classical baseline.
- State the main qualification: this does not compare against Spotify's production recommender.

## Introduction
- Hook: users can now export listening history and ask a model directly.
- Gap: prior LLM4Rec work is mostly hybrid or fine-tuned, not prompt-only music recommendation.
- Approach: same 20-item candidate sets for LLMs and classical baselines across two public tasks.
- Quantitative preview: GPT-4.1 minimal reaches 0.589 MRR on Last.fm and 0.329 on MPD.
- Contributions:
  - controlled prompt-only comparison
  - two public music tasks
  - prompt ablation and statistical tests
  - boundary on the practical claim

## Related Work
- LLMs as recommenders and instruction followers.
- Hybrid LLM+recommender systems.
- Music recommendation and playlist continuation baselines.
- Position our work against hybrid and prompt-only settings.

## Methodology
- Problem setup with history, candidate set, and single relevant target.
- Datasets and example construction.
- Baselines and LLM prompt conditions.
- Metrics, statistical testing, environment, and cost.
- Planning figure: pipeline from history to candidate ranking.

## Results
- Last.fm table.
- MPD table.
- Figures with RR and NDCG plots for both datasets.
- Primary significance claims with corrected p-values and effect sizes.
- Prompt ablation and model comparison summary.

## Discussion
- Why minimal prompts likely work better.
- Popularity bias and coverage interpretation.
- Representative success/failure modes.
- Limitations and broader implications.

## Conclusion
- Restate contribution and narrow claim.
- Main quantitative finding.
- Future work on stronger baselines, open-ended recommendation, and human evaluation.
