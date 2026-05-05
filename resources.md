# Resources Catalog

## Summary

This catalog lists the papers, datasets, and repositories gathered for the project on whether LLMs can solve music recommendation as well as or better than Spotify-style algorithms.

## Papers

Total papers downloaded: 11

| Title | Authors | Year | File | Key Info |
|------|---------|------|------|----------|
| A Survey on Large Language Models for Recommendation | Wu et al. | 2023 | `papers/2305.19860_survey_llm_recommendation.pdf` | Taxonomy of LLM4Rec methods |
| Recommender Systems in the Era of Large Language Models (LLMs) | Zhao et al. | 2023 | `papers/2307.02046_era_llm_recommender_systems.pdf` | Survey of prompting/fine-tuning/pre-training |
| Recommendation as Language Processing (P5) | Geng et al. | 2022 | `papers/2203.13366_p5_recommendation_as_language_processing.pdf` | Foundational text-to-text recommendation |
| Recommendation as Instruction Following | Zhang et al. | 2023 | `papers/2305.07001_recommendation_as_instruction_following.pdf` | Instruction-tuned open LLM recommender |
| LLM-Rec | Lyu et al. | 2023 | `papers/2307.15780_llm_rec_personalized_recommendation.pdf` | LLM prompting for text enrichment |
| LLaRA | Liao et al. | 2023 | `papers/2312.02445_llara_large_language_recommendation_assistant.pdf` | Hybrid sequential recommender with LastFM path |
| A-LLMRec | Kim et al. | 2024 | `papers/2404.11343_a_llmrec_collaborative_filtering.pdf` | Hybrid CF+LLM model with strong warm/cold performance |
| LLMs as Recommenders: Popularity Bias | Lichtenberg et al. | 2024 | `papers/2406.01285_llms_as_recommenders_popularity_bias.pdf` | Accuracy must be complemented with bias analysis |
| TALKPLAY | Doh et al. | 2025 | `papers/2502.13713_talkplay_multimodal_music_recommendation.pdf` | Most relevant recent music-specific LLM paper |
| Automatic Playlist Continuation through a Composition of Collaborative Filters | Teinemaa et al. | 2018 | `papers/1808.04288_automatic_playlist_continuation_collaborative_filters.pdf` | Strong classical Spotify MPD baseline |
| Analysis of the ACM RecSys Challenge 2018 | Zamani et al. | 2018 | `papers/1810.01520_analysis_recsys_challenge_2018_playlist_continuation.pdf` | Official benchmark framing and results |

See `papers/README.md` for more detail.

## Datasets

Total dataset artifacts downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Last.fm 1K subset | Hugging Face | 120k interactions locally; full dataset ~469 MB download / ~3.4 GB materialized | Sequential music recommendation | `datasets/lastfm_1k_subset/` | Best immediately usable public listening-history dataset gathered here |
| Spotify MPD sample | Hugging Face mirror of Spotify MPD | One JSON slice | Playlist continuation | `datasets/spotify_mpd_sample/` | Good for schema/prototyping; full MPD recommended for benchmark-scale evaluation |

See `datasets/README.md` for download instructions and notes.

## Code Repositories

Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| A-LLMRec | https://github.com/ghdtjr/A-LLMRec | CF+LLM hybrid recommender | `code/A-LLMRec/` | Strong template for hybrid recommendation |
| LLaRA | https://github.com/ljy0ustc/LLaRA | Sequential LLM recommender | `code/LLaRA/` | Includes LastFM-specific train/test scripts |
| RecBole | https://github.com/RUCAIBox/RecBole | Baseline benchmark framework | `code/RecBole/` | Broadest reusable baseline library |
| recsys-smpd | https://github.com/anaezquerro/recsys-smpd | Spotify MPD classical baselines | `code/recsys-smpd/` | Directly relevant Spotify playlist-continuation code |

See `code/README.md` for more detail.

## Resource Gathering Notes

### Search Strategy

- Started with the local `paper-finder` helper in diligent mode.
- The helper service stalled, so manual primary-source search was used.
- Prioritized recent arXiv papers for LLM4Rec, then older Spotify MPD papers for strong music baselines.
- Preferred public datasets with clear loading paths and repositories with reusable code.

### Selection Criteria

- Direct relevance to LLM-based recommendation or music recommendation.
- Availability of PDF, dataset, or code.
- Utility for building a rigorous experiment rather than just a narrative review.
- Balance between domain-specific resources (Spotify/LastFM/music) and stronger generic LLM4Rec methods.

### Challenges Encountered

- The local paper-finder backend did not return usable results in time.
- Full public Spotify-like datasets are fragmented or large.
- The cleanest public music-listening datasets are LastFM-based rather than Spotify export data.

### Gaps and Workarounds

- No direct public substitute exactly matches a private Spotify export.
- Workaround: use LastFM for fast iteration, MPD for Spotify-like public evaluation, and real Spotify export data for final external validity.

## Recommendations for Experiment Design

1. Primary datasets: use `lastfm_1k_subset` for first-pass local development, then move to full LastFM and Spotify MPD; if available, finish with personal Spotify export data.
2. Baseline methods: popularity, item-kNN, PureSVD, SASRec or GRU4Rec, BM25 for textual query matching, and one hybrid LLM model pattern inspired by A-LLMRec or LLaRA.
3. Evaluation metrics: NDCG@K, Recall@K, Hit@K, MRR, R-precision for playlist tasks, plus novelty/diversity and popularity bias.
4. Code to adapt/reuse: use `RecBole` for classical baselines, `recsys-smpd` for Spotify playlist continuation, and `LLaRA` as the most directly adaptable LLM-music bridge.
