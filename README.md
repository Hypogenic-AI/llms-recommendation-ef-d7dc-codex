# LLM Music Recommendation Study

This project tests a practical question: if you export your Spotify-style listening data and ask a frontier LLM for music recommendations, can it beat classical recommendation algorithms on the same task? The study evaluates prompt-only `gpt-4.1` and `Claude Sonnet 4.5` against popularity and co-occurrence baselines on public Last.fm and Spotify MPD tasks.

Key findings:
- Prompt-only LLM ranking beat both classical baselines on both tasks.
- The strongest condition was the simple history-only prompt, not the richer profile prompt.
- Claude and GPT performed similarly; neither had a reliable edge over the other.
- The LLM outputs were less popularity-concentrated and slightly higher-coverage than the classical baselines.
- This does **not** prove that prompting Claude beats Spotify’s internal recommender; it shows that prompt-only LLMs can beat lightweight public Spotify-style baselines under a controlled offline protocol.

Full details are in [REPORT.md](./REPORT.md).

## Reproduce

```bash
source .venv/bin/activate
python src/music_recs_experiment.py \
  --lastfm-examples 50 \
  --mpd-examples 50 \
  --candidate-size 20 \
  --output-dir results \
  --figures-dir figures
```

## File Structure

- `planning.md`: research plan and hypothesis decomposition
- `src/music_recs_experiment.py`: main experiment pipeline
- `prompts/`: prompt templates used for minimal and profile conditions
- `results/`: cached API outputs, per-example metrics, summaries, significance tests
- `figures/`: metric comparison plots for both datasets
- `literature_review.md`: synthesized background review
- `resources.md`: catalog of gathered papers, datasets, and code

## Notes

- Environment: Python `3.12.8` in the local `.venv`
- Models: `gpt-4.1` and `anthropic/claude-sonnet-4.5`
- Execution date: `2026-05-05`
