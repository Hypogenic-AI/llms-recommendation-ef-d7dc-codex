# Research Plan

## Motivation & Novelty Assessment

### Why This Research Matters
Users increasingly have access to rich personal listening-history exports, and current LLMs can reason over that history in natural language without any bespoke recommender training. If prompt-only LLM recommendation were competitive with strong Spotify-style ranking baselines, users could obtain high-quality recommendations from a model like Claude using only exported data and a text prompt rather than depending entirely on a platform recommender.

### Gap in Existing Work
The gathered literature shows that LLMs can help recommendation, but the strongest reported wins usually come from hybrid systems such as A-LLMRec, LLaRA, or TALKPLAY rather than from prompt-only recommendation over user history. Public evidence on music recommendation is thinner than for movies or e-commerce, and there is little direct evidence that a general-purpose LLM prompted with listening history can beat strong Spotify-like collaborative or playlist-continuation baselines.

### Our Novel Contribution
This study directly tests the user's practical question: given only exported listening/playlist history and no task-specific fine-tuning, can a frontier LLM produce better music recommendations than classical Spotify-style algorithms? The novelty is the controlled head-to-head comparison between prompt-only LLM ranking and classical ranking baselines on the same candidate sets across two public music tasks: sequential next-track prediction and playlist continuation.

### Experiment Justification
- Experiment 1: Last.fm next-track ranking. This tests the closest public analogue to a personal Spotify export by using chronological user listening histories.
- Experiment 2: Spotify MPD playlist continuation. This tests a Spotify-native public benchmark setting where recommendation is based on playlist context rather than a live user session.
- Experiment 3: Prompt ablation within the LLM condition. This tests whether richer profile summaries materially improve prompt-only recommendation quality over a minimal history-only prompt.

## Research Question
Can a prompt-only frontier LLM, given music listening or playlist history and a fixed candidate pool, generate recommendations that match or outperform classical Spotify-style recommendation baselines?

## Background and Motivation
Prior work in `literature_review.md` indicates that LLMs can add semantic reasoning and conversational flexibility to recommendation, but the strongest quantitative results usually come from hybrid systems that combine collaborative filtering or sequential models with LLM components. Music-specific evidence is limited, and the most relevant strong classical baselines still come from Spotify Million Playlist Dataset work and sequence-aware recommenders. This study fills the gap between narrative claims about "asking Claude for recommendations" and reproducible offline ranking evidence.

## Hypothesis Decomposition
- H1: Prompt-only LLM ranking can beat a non-personalized popularity baseline on music recommendation tasks.
- H2: Prompt-only LLM ranking can match or beat a stronger personalized/co-occurrence Spotify-style baseline on at least one public task.
- H3: A richer prompt containing a compact user profile and recent history will outperform a minimal history-only prompt.
- H4: Even if the LLM underperforms on accuracy, it may produce less popularity-biased recommendations than classical baselines.

Independent variables:
- Recommendation method: popularity, co-occurrence/item-kNN-style baseline, LLM minimal prompt, LLM profile prompt.
- Task type: Last.fm next-track ranking, Spotify MPD playlist continuation.

Dependent variables:
- Hit@10
- NDCG@10
- MRR
- Mean candidate popularity rank
- Catalog coverage / distinct recommended items

Alternative explanations:
- LLM failure may reflect missing metadata such as audio features, genres, or embeddings rather than a fundamental inability to recommend.
- Candidate-set ranking may underestimate conversational value because the LLM is constrained to a closed pool.
- Public datasets may differ from a real Spotify export in richness and temporal recency.

## Proposed Methodology

### Approach
Use an offline candidate-ranking protocol so every method solves the same problem. For each example, construct a candidate set containing one ground-truth relevant track and multiple negatives. Classical baselines score the candidates numerically; LLMs receive the same candidate set plus history context and must rank the options. This avoids unfair open-ended generation and enables direct comparison with standard ranking metrics.

### Experimental Steps
1. Inspect and validate the local Last.fm subset and Spotify MPD sample.
   Rationale: confirm schema, sample sizes, and whether both datasets support a consistent ranking protocol.
2. Build evaluation examples.
   Rationale: create held-out candidate-ranking tasks from user histories and playlists with reproducible negative sampling.
3. Implement classical baselines.
   Rationale: establish a floor and a stronger Spotify-style personalized baseline before invoking any LLM.
4. Implement LLM ranking prompts and API harness.
   Rationale: test the practical "ask Claude with export data" workflow using real model calls and cached responses.
5. Run both tasks under identical candidate pools.
   Rationale: produce apples-to-apples comparisons and prompt ablations.
6. Perform statistical analysis and error analysis.
   Rationale: determine whether any gains are reliable and understand where each method succeeds or fails.

### Baselines
- Popularity baseline: rank candidate items by training-set frequency.
- Personalized co-occurrence baseline: score candidates by co-occurrence with history items and artist continuity, approximating a lightweight Spotify-style collaborative heuristic.
- LLM minimal prompt: frontier LLM receives recent listening/playlist context and candidate list only.
- LLM profile prompt: frontier LLM receives recent context plus compact profile statistics such as top artists and countries when available.

### Evaluation Metrics
- Hit@10: whether the true held-out item appears in the top 10.
- NDCG@10: ranking quality with stronger weight on higher ranks.
- MRR: reciprocal rank of the first relevant item; useful for single-target candidate ranking.
- Mean popularity percentile of recommended top-10 items: proxy for popularity bias.
- Coverage: number of unique items appearing in top-10 outputs across the test set.

These metrics align with the literature review recommendations and support both accuracy and beyond-accuracy analysis.

### Statistical Analysis Plan
- Primary hypothesis test: paired Wilcoxon signed-rank tests on per-example reciprocal rank and NDCG differences because example-level metric distributions are typically non-normal.
- Confidence intervals: bootstrap 95% confidence intervals over test examples for aggregate metrics.
- Effect size: paired Cohen's d on per-example metric differences.
- Multiple comparisons: Benjamini-Hochberg correction across primary pairwise comparisons within each dataset.
- Significance level: alpha = 0.05.

## Expected Outcomes
Results that would support the strongest form of the hypothesis:
- An LLM condition significantly exceeds the personalized baseline on NDCG@10 or MRR on at least one task.

Results that would partially support the hypothesis:
- The LLM beats popularity but not the personalized baseline, while offering lower popularity bias or better coverage.

Results that would refute the practical claim:
- The LLM fails to beat the personalized baseline on either task and only matches or barely beats popularity.

## Timeline and Milestones
1. Planning and dataset verification: 20-30 minutes.
2. Implementation of evaluation pipeline and baselines: 45-60 minutes.
3. API integration and prompt evaluation: 45-60 minutes.
4. Analysis, figures, and write-up: 30-45 minutes.
5. Validation and reproducibility check: 15-20 minutes.

## Potential Challenges
- The local Last.fm subset may be sparse for many users.
  Mitigation: require minimum history length and cap examples per user.
- The MPD sample is only one slice, not the full benchmark.
  Mitigation: treat MPD as a small external-validity check and document the limitation.
- LLM output may be malformed or omit candidates.
  Mitigation: require JSON output, validate responses, and retry invalid generations with cached logging.
- API cost or latency may become large.
  Mitigation: use a bounded candidate set, cache all responses, and start with a smaller development subset before scaling.

## Success Criteria
- A reproducible pipeline exists under `src/` and saves results under `results/`.
- At least one real frontier LLM is evaluated with actual API calls.
- Both public music tasks are executed with the same ranking protocol.
- REPORT.md contains actual numerical results, figures, and a clear answer to the user's question.
