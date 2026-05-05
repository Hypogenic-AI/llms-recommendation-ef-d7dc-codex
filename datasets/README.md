# Downloaded Datasets

This directory contains locally downloaded dataset artifacts for the recommendation experiments. Large data files are intentionally excluded from git.

## Dataset 1: Last.fm 1K Subset

- Source: `matthewfranglen/lastfm-1k` on Hugging Face
- Local artifact: `datasets/lastfm_1k_subset/`
- Format: Hugging Face `DatasetDict`
- Task fit: sequential and personalized music recommendation
- Local subset size: 100,000 train interactions, 10,000 validation interactions, 10,000 test interactions
- Available fields: `user_id`, `artist_name`, `track_name`, `timestamp`, `country`, plus integer indices and MusicBrainz IDs
- Why useful: closest immediately accessible public dataset to "Spotify export"-style listening history

### Download Instructions

Full dataset materialization:

```python
from datasets import load_dataset
dataset = load_dataset("matthewfranglen/lastfm-1k")
dataset.save_to_disk("datasets/lastfm_1k")
```

Re-create the smaller local subset used in this workspace:

```python
from datasets import load_dataset, DatasetDict

subset = DatasetDict({
    "train": load_dataset("matthewfranglen/lastfm-1k", split="train[:100000]"),
    "valid": load_dataset("matthewfranglen/lastfm-1k", split="valid[:10000]"),
    "test": load_dataset("matthewfranglen/lastfm-1k", split="test[:10000]"),
})
subset.save_to_disk("datasets/lastfm_1k_subset")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/lastfm_1k_subset")
```

### Notes

- Hugging Face metadata reports the full `lastfm-1k` download size at about 469 MB and dataset size at about 3.4 GB.
- This is the best public music-listening dataset gathered here for fast local experimentation.
- It is not Spotify data, so it is best for prototyping rather than final hypothesis testing.

## Dataset 2: Spotify Million Playlist Dataset Sample

- Source: `jaxliu/Spotify_Million_Playlist_Dataset_Challenge` on Hugging Face
- Local artifact: `datasets/spotify_mpd_sample/data/mpd.slice.0-999.json`
- Format: raw JSON slice from the Spotify Million Playlist Dataset mirror
- Task fit: playlist continuation and playlist-based recommendation
- Why useful: domain match with Spotify playlists and challenge-style evaluation

### Download Instructions

Single-slice sample:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="jaxliu/Spotify_Million_Playlist_Dataset_Challenge",
    repo_type="dataset",
    filename="data/mpd.slice.0-999.json",
    local_dir="datasets/spotify_mpd_sample",
    local_dir_use_symlinks=False,
)
```

Full mirror download:

```bash
git lfs install
git clone https://huggingface.co/datasets/jaxliu/Spotify_Million_Playlist_Dataset_Challenge datasets/spotify_mpd_full
```

### Loading

```python
import json

with open("datasets/spotify_mpd_sample/data/mpd.slice.0-999.json") as f:
    sample = json.load(f)
```

### Notes

- The full MPD is closer to the Spotify recommendation setting than Last.fm.
- This sample is useful for parser development and schema checks.
- For stronger evaluation, pair MPD or real Spotify export data with a candidate-generation baseline and an LLM reranker or generator.

## Recommended Additional Data

- Personal Spotify export data: best match for the hypothesis, but not directly downloadable by this agent.
- Official MPD or mirrored full MPD: best public Spotify-like benchmark for playlist completion.
