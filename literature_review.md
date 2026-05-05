# Literature Review: How much do LLMs solve recommendation algorithms?

## Review Scope

### Research Question
Can LLMs generate music recommendations from user listening history or playlist context that match or outperform classical Spotify-like recommenders?

### Inclusion Criteria

- Paper studies LLMs for recommendation, sequential recommendation, conversational recommendation, or music recommendation.
- Paper provides methodological detail or code relevant to experimentation.
- Paper is directly relevant to Spotify-style playlist/listening-history recommendation or strong enough to inform baseline design.

### Exclusion Criteria

- Purely general LLM papers without recommendation content.
- Music papers without a recommendation component.
- Non-public resources with no usable methodological detail.

### Time Frame
2018-2025, with emphasis on 2023-2025 LLM work and 2018 Spotify MPD baselines.

### Sources

- arXiv
- Hugging Face datasets
- GitHub repositories
- ACM RecSys challenge materials

## Search Log

| Date | Query | Source | Notes |
|------|-------|--------|-------|
| 2026-05-05 | LLM recommendation systems music recommendation Spotify data | local `paper-finder` + manual fallback | Helper service stalled, so manual primary-source search was used |
| 2026-05-05 | LLM recommendation survey / A-LLMRec / LLaRA / TalkPlay | arXiv | Gathered recent LLM4Rec papers |
| 2026-05-05 | Spotify Million Playlist Dataset / Last.fm datasets | Hugging Face + official challenge pages | Gathered public music datasets |

## Key Papers

### A Survey on Large Language Models for Recommendation

- Authors: Likang Wu et al.
- Year: 2023
- Source: arXiv 2305.19860
- Key contribution: organizes LLM recommenders into discriminative and generative families.
- Methodology: survey and taxonomy rather than a new model.
- Datasets used: summarizes common datasets such as MovieLens, Amazon, LastFM.
- Results: identifies prompting, fine-tuning, and representation-learning patterns.
- Code available: yes, paper list repo referenced in the paper.
- Relevance: useful map of the design space, but not evidence that LLMs alone beat mature recommenders.

### Recommender Systems in the Era of Large Language Models (LLMs)

- Authors: Zihuai Zhao et al.
- Year: 2023
- Source: arXiv 2307.02046
- Key contribution: frames three paradigms for using LLMs in recommendation: pre-training, fine-tuning, and prompting.
- Methodology: broad survey of recommendation-enhancement patterns.
- Datasets used: overview paper, not one benchmark.
- Results: argues LLMs add semantic understanding and reasoning, but data sparsity and evaluation remain open issues.
- Code available: no central experimental repo.
- Relevance: good high-level framing for a hybrid study rather than an LLM-only study.

### Recommendation as Language Processing (P5)

- Authors: Shijie Geng et al.
- Year: 2022
- Source: arXiv 2203.13366
- Key contribution: converts recommendation tasks into text-to-text generation.
- Methodology: unified prompt-and-predict framework over multiple recommendation tasks.
- Datasets used: several recommendation benchmarks across tasks.
- Results: showed that language-style reformulation can work before current instruction-tuned LLMs.
- Code available: yes.
- Relevance: foundational precursor for treating recommendation as generation.

### Recommendation as Instruction Following

- Authors: Junjie Zhang et al.
- Year: 2023
- Source: arXiv 2305.07001
- Key contribution: instruction-tunes Flan-T5-XL with 39 recommendation templates and 252K generated instructions.
- Methodology: natural-language instructions represent preference, context, and task type.
- Datasets used: several real-world recommendation/search datasets.
- Baselines: classical recommenders and GPT-3.5.
- Results: reports wins over several strong baselines including GPT-3.5 on evaluation tasks.
- Code available: paper reports open-source model path.
- Relevance: strong evidence that task-specific instruction tuning helps, but it is not music-specific.

### LLM-Rec: Personalized Recommendation via Prompting Large Language Models

- Authors: Hanjia Lyu et al.
- Year: 2023
- Source: arXiv 2307.15780
- Key contribution: uses LLM prompting to enrich item text and improve downstream recommendation.
- Methodology: four prompting strategies for text enrichment; recommendation is still done by simpler downstream models.
- Datasets used: text-rich recommendation benchmarks.
- Results: LLM-enriched text improves quality and allows simple MLP models to compete with stronger content methods.
- Code available: yes.
- Relevance: useful for a Spotify study because it suggests LLMs may be more valuable as feature generators than as standalone recommenders.

### LLaRA: Large Language-Recommendation Assistant

- Authors: Jiayi Liao et al.
- Year: 2023
- Source: arXiv 2312.02445
- Key contribution: aligns sequential recommender ID embeddings with LLM inputs using a projector and hybrid prompts.
- Methodology: combines traditional recommender embeddings plus text; curriculum from text-only to hybrid prompts.
- Datasets used: MovieLens, Steam, LastFM.
- Results: repo reports best HitRatio@1 among its variants on LastFM with SASRec backbone.
- Code available: yes, including LastFM scripts.
- Relevance: directly useful because it already has a LastFM path and is closer to music recommendation than many generic LLM4Rec papers.

### A-LLMRec

- Authors: Sein Kim et al.
- Year: 2024
- Source: arXiv 2404.11343 / KDD 2024
- Key contribution: injects collaborative knowledge from a frozen CF recommender into a frozen LLM in a two-stage alignment pipeline.
- Methodology: Stage 1 aligns item embeddings with text via SBERT and MLPs; Stage 2 projects joint knowledge into the LLM.
- Datasets used: Amazon categories in the public paper and repo.
- Baselines: SASRec, MoRec, CTRL, RECFORMER, TALLRec, LLM-only variants.
- Results: deep-read tables show strong Hit@1 gains in both cold and warm item settings over the listed baselines.
- Code available: yes.
- Relevance: the clearest evidence in this set that hybrid CF+LLM approaches outperform LLM-only approaches.

### Large Language Models as Recommender Systems: A Study of Popularity Bias

- Authors: Jan Malte Lichtenberg et al.
- Year: 2024
- Source: arXiv 2406.01285
- Key contribution: evaluates LLM-based recommendation through the lens of popularity bias.
- Methodology: compares a simple LLM recommender against traditional methods on a movie task and proposes a bias metric.
- Datasets used: movie recommendation data.
- Results: LLM recommender shows less popularity bias in that setup.
- Code available: not identified in this pass.
- Relevance: important warning that better recommendation is not only accuracy; novelty and bias matter, especially for music.

### TALKPLAY: Multimodal Music Recommendation with Large Language Models

- Authors: Seungheon Doh, Keunwoo Choi, Juhan Nam
- Year: 2025
- Source: arXiv 2502.13713
- Key contribution: reframes conversational music recommendation as token generation over a combined text-plus-music vocabulary.
- Methodology: multimodal music tokenizer over audio, lyrics, metadata, semantic tags, and playlist co-occurrence; supervised fine-tuning of an LLM; generation of music tokens before natural-language response.
- Datasets used: synthetic music recommendation conversations and multimodal music metadata.
- Baselines: BM25, CLAP-Music, NV-Embed-V2, SASRec, TIGER.
- Results: deep-read Table 2 shows TALKPLAY best overall, with MRR 0.049, Hit@1 0.026, Hit@100 0.288, and major gains over the strongest baseline in Hit@1.
- Code available: demo available; code availability not clearly resolved from the PDF chunk pass.
- Relevance: the strongest direct evidence gathered here for LLM-based music recommendation.

### Spotify MPD Baseline Papers

- 2018 challenge papers show what a strong non-LLM baseline looks like for Spotify-style music recommendation.
- `Automatic Playlist Continuation through a Composition of Collaborative Filters` combines playlist-title, artist, album, and track collaborative filters and reached 12th/112 on the challenge leaderboard.
- `An Analysis of Approaches Taken in the ACM RecSys Challenge 2018` reports best main-track metrics around R-precision 0.2241 and NDCG 0.3946 and shows how strong hybrid or neighborhood-style methods can be.
- Relevance: these are the baseline floor that an LLM system must beat on Spotify-like public data.

## Common Methodologies

- LLM as generator: P5, instruction-following recommendation, TALKPLAY.
- LLM as semantic augmenter: LLM-Rec.
- Hybrid recommender + LLM alignment: LLaRA and A-LLMRec.
- Classical music baselines: collaborative filtering, neighborhood models, PureSVD, track2vec, SASRec.

## Standard Baselines

- Popularity baseline: easy but weak; still essential.
- Item-kNN / user-kNN / neighborhood methods: strong classical baselines on playlist continuation.
- PureSVD / matrix factorization: compact collaborative baseline.
- SASRec or GRU4Rec: standard sequential neural baselines.
- BM25 / dense retrieval: useful when recommendation is driven by explicit natural-language music queries.

## Evaluation Metrics

- Ranking quality: Hit@K, Recall@K, NDCG@K, MRR.
- Playlist continuation metrics: R-precision, NDCG, clicks.
- Beyond accuracy: popularity bias, novelty, diversity, and conversational response quality.

## Datasets in the Literature

- Spotify Million Playlist Dataset: standard public benchmark for playlist continuation.
- LastFM: useful for sequential music recommendation and user listening histories.
- Amazon / MovieLens / Steam: common in LLM4Rec papers but weaker domain match for this hypothesis.

## Gaps and Opportunities

- Most LLM4Rec papers are not music-specific.
- Many strong LLM results are hybrid, not pure LLM-only systems.
- Public evidence against strong Spotify-style baselines is still limited.
- Personal Spotify export data is the best domain match but raises evaluation and candidate-set design issues.

## Recommendations for Our Experiment

- Recommended datasets: full `lastfm-1k` or `lastfm-360k` for rapid prototyping; Spotify MPD for public Spotify-like evaluation; personal Spotify export for final hypothesis testing.
- Recommended baselines: popularity, item-kNN, PureSVD, SASRec/GRU4Rec, and a non-LLM retrieval baseline such as BM25 when using textual prompts.
- Recommended metrics: Hit@10, Recall@10, NDCG@10, MRR, novelty/diversity, and popularity-bias metrics.
- Methodological considerations:
  - Do not compare an unconstrained LLM against Spotify recommendations without fixing the candidate set.
  - Test hybrid systems, not just pure prompting.
  - Use the same held-out next-track or held-out playlist-completion protocol for all methods.
  - Separate "recommendation quality" from "response quality"; TALKPLAY suggests these can be optimized together, but many baselines cannot.
