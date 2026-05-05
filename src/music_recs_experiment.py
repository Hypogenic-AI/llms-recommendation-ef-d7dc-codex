"""Offline evaluation of prompt-only LLM music recommendation.

This script builds two public-data tasks:
1. Last.fm next-track ranking from user listening histories.
2. Spotify MPD playlist continuation from playlist prefixes.

Each example uses a fixed candidate pool containing one positive item and
multiple negatives. Classical baselines and LLMs rank the same candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from datasets import concatenate_datasets, load_dataset, load_from_disk
from scipy.stats import wilcoxon


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class Example:
    dataset: str
    example_id: str
    context_type: str
    context_title: str | None
    user_id: str
    history: list[str]
    history_artists: list[str]
    top_artists: list[str]
    country: str | None
    target_id: str
    target_label: str
    candidates: list[dict[str, str]]


class LLMClient:
    def __init__(self, cache_path: Path, openai_model: str, claude_model: str) -> None:
        self.cache_path = cache_path
        self.openai_model = openai_model
        self.claude_model = claude_model
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        cache: dict[str, dict[str, Any]] = {}
        with self.cache_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                cache[row["cache_key"]] = row
        return cache

    def _append_cache(self, row: dict[str, Any]) -> None:
        self.cache[row["cache_key"]] = row
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def rank_candidates(
        self,
        provider: str,
        prompt_name: str,
        example: Example,
        prompt: str,
        candidate_ids: list[str],
        temperature: float = 0.0,
    ) -> list[str]:
        cache_key = f"{provider}|{prompt_name}|{example.example_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]["ranked_ids"]

        if provider == "openai":
            model = self.openai_model
            response_text, meta = self._call_openai(model, prompt, temperature)
        elif provider == "claude":
            model = self.claude_model
            response_text, meta = self._call_openrouter(model, prompt, temperature)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        ranked_ids = parse_ranked_ids(response_text, candidate_ids)
        row = {
            "cache_key": cache_key,
            "provider": provider,
            "model": model,
            "prompt_name": prompt_name,
            "example_id": example.example_id,
            "response_text": response_text,
            "ranked_ids": ranked_ids,
            "meta": meta,
            "timestamp": time.time(),
        }
        self._append_cache(row)
        return ranked_ids

    def _call_openai(self, model: str, prompt: str, temperature: float) -> tuple[str, dict[str, Any]]:
        client = requests.Session()
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "input": prompt,
            "temperature": temperature,
        }
        response = client.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        output_text = body.get("output_text")
        if not output_text:
            output_parts = []
            for output in body.get("output", []):
                for content in output.get("content", []):
                    if content.get("type") == "output_text":
                        output_parts.append(content.get("text", ""))
            output_text = "\n".join(output_parts).strip()
        meta = {
            "id": body.get("id"),
            "usage": body.get("usage"),
        }
        return output_text, meta

    def _call_openrouter(self, model: str, prompt: str, temperature: float) -> tuple[str, dict[str, Any]]:
        client = requests.Session()
        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_KEY']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        response = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"]["content"]
        meta = {
            "id": body.get("id"),
            "usage": body.get("usage"),
        }
        return message, meta


def parse_ranked_ids(response_text: str, candidate_ids: list[str]) -> list[str]:
    candidate_set = set(candidate_ids)
    try:
        body = json.loads(response_text)
        ranked = body.get("ranked_ids") or body.get("ranking") or []
        ranked = [item for item in ranked if item in candidate_set]
    except json.JSONDecodeError:
        ranked = re.findall(r"C\d{2}", response_text)
        ranked = [item for item in ranked if item in candidate_set]

    deduped: list[str] = []
    seen = set()
    for item in ranked:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    for item in candidate_ids:
        if item not in seen:
            deduped.append(item)
    return deduped


def track_label(artist: str, track: str) -> str:
    return f"{artist} - {track}"


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def build_lastfm_examples(max_examples: int, candidate_size: int, seed: int) -> tuple[list[Example], dict[str, Any]]:
    dataset = load_from_disk("datasets/lastfm_1k_subset")
    full_df = concatenate_datasets([dataset["train"], dataset["valid"], dataset["test"]]).to_pandas()
    if full_df["user_id"].nunique() < 50:
        remote_ds = load_dataset("matthewfranglen/lastfm-1k", split="train[:1000000]")
        full_df = remote_ds.to_pandas()
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], utc=True, errors="coerce")
    full_df = full_df.dropna(subset=["timestamp", "artist_name", "track_name", "user_id"])
    full_df["track_id"] = full_df["artist_name"].astype(str) + " || " + full_df["track_name"].astype(str)
    full_df = full_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    grouped = full_df.groupby("user_id", sort=False)
    eligible_users = [user_id for user_id, frame in grouped if len(frame) >= 12]
    rng = random.Random(seed)
    rng.shuffle(eligible_users)

    popularity = Counter(full_df["track_id"])
    track_to_artist = (
        full_df[["track_id", "artist_name", "track_name"]]
        .drop_duplicates("track_id")
        .set_index("track_id")
        .to_dict("index")
    )
    candidate_pool = [track for track, _ in popularity.most_common(5000)]

    examples: list[Example] = []
    train_sequences: list[list[str]] = []
    train_artist_sequences: list[list[str]] = []

    pending_examples: list[dict[str, Any]] = []
    for user_id in eligible_users:
        user_df = grouped.get_group(user_id).copy()
        history_df = user_df.iloc[:-1]
        target_row = user_df.iloc[-1]
        if len(history_df) < 10:
            continue

        train_sequences.append(history_df["track_id"].tolist())
        train_artist_sequences.append(history_df["artist_name"].astype(str).tolist())

        history_tail = history_df.tail(10)
        history_tracks = history_tail["track_id"].tolist()
        history_labels = [track_label(row.artist_name, row.track_name) for row in history_tail.itertuples()]
        top_artists = [artist for artist, _ in Counter(history_df["artist_name"].astype(str)).most_common(5)]
        used = set(history_df["track_id"])

        pending_examples.append(
            {
                "user_id": user_id,
                "history_labels": history_labels,
                "history_artists": history_tail["artist_name"].astype(str).tolist(),
                "top_artists": top_artists,
                "country": target_row.get("country"),
                "target_id": target_row["track_id"],
                "target_label": track_label(target_row["artist_name"], target_row["track_name"]),
                "used": used,
            }
        )
        if len(pending_examples) >= max_examples * 3:
            break

    for row in pending_examples:
        top_artist_set = set(row["top_artists"])
        artist_matched = [
            track
            for track in candidate_pool
            if track not in row["used"]
            and track != row["target_id"]
            and track_to_artist[track]["artist_name"] in top_artist_set
        ]
        semihard = [
            track
            for track in candidate_pool
            if track not in row["used"]
            and track != row["target_id"]
            and track not in artist_matched
            and popularity[track] >= 2
        ]
        negatives = artist_matched[: max(4, candidate_size // 2)] + semihard[: candidate_size - 1]
        negatives = negatives[: candidate_size - 1]
        if len(negatives) < candidate_size - 1:
            continue

        candidate_tracks = negatives + [row["target_id"]]
        rng.shuffle(candidate_tracks)
        candidates = []
        for idx, track_id in enumerate(candidate_tracks, start=1):
            meta = track_to_artist[track_id]
            candidates.append(
                {
                    "candidate_id": f"C{idx:02d}",
                    "track_id": track_id,
                    "artist_name": meta["artist_name"],
                    "track_name": meta["track_name"],
                    "label": track_label(meta["artist_name"], meta["track_name"]),
                }
            )
        examples.append(
            Example(
                dataset="lastfm",
                example_id=f"lastfm::{row['user_id']}",
                context_type="recent_listens",
                context_title=None,
                user_id=row["user_id"],
                history=row["history_labels"],
                history_artists=row["history_artists"],
                top_artists=row["top_artists"],
                country=row["country"],
                target_id=row["target_id"],
                target_label=row["target_label"],
                candidates=candidates,
            )
        )
        if len(examples) >= max_examples:
            break

    metadata = {
        "num_events": int(len(full_df)),
        "num_users": int(full_df["user_id"].nunique()),
        "eligible_users": int(len(eligible_users)),
        "candidate_pool_size": int(len(candidate_pool)),
        "train_sequence_count": len(train_sequences),
        "track_popularity": dict(popularity.most_common(20)),
        "track_to_artist": track_to_artist,
        "train_sequences": train_sequences,
        "train_artist_sequences": train_artist_sequences,
    }
    return examples, metadata


def build_mpd_examples(max_examples: int, candidate_size: int, seed: int) -> tuple[list[Example], dict[str, Any]]:
    with open("datasets/spotify_mpd_sample/data/mpd.slice.0-999.json") as handle:
        data = json.load(handle)
    playlists = data["playlists"]
    playlists = [playlist for playlist in playlists if len(playlist["tracks"]) >= 8]

    def to_track_id(track: dict[str, Any]) -> str:
        return track["track_uri"]

    track_meta: dict[str, dict[str, str]] = {}
    popularity: Counter[str] = Counter()
    train_sequences: list[list[str]] = []
    for playlist in playlists:
        seq = []
        for track in playlist["tracks"][:-1]:
            track_id = to_track_id(track)
            seq.append(track_id)
            popularity[track_id] += 1
            track_meta[track_id] = {
                "artist_name": track["artist_name"],
                "track_name": track["track_name"],
            }
        train_sequences.append(seq)

    rng = random.Random(seed)
    candidate_pool = [track_id for track_id, _ in popularity.most_common(10000)]
    examples: list[Example] = []
    for playlist in playlists:
        prefix = playlist["tracks"][:-1]
        target = playlist["tracks"][-1]
        history_ids = [to_track_id(track) for track in prefix[-10:]]
        used = {to_track_id(track) for track in prefix}
        target_id = to_track_id(target)
        if target_id not in track_meta:
            track_meta[target_id] = {
                "artist_name": target["artist_name"],
                "track_name": target["track_name"],
            }

        prefix_artists = [track["artist_name"] for track in prefix]
        top_artists = [artist for artist, _ in Counter(prefix_artists).most_common(5)]
        same_artist = [
            track_id
            for track_id in candidate_pool
            if track_id not in used and track_id != target_id and track_meta[track_id]["artist_name"] in set(top_artists)
        ]
        others = [
            track_id
            for track_id in candidate_pool
            if track_id not in used and track_id != target_id and track_id not in same_artist
        ]
        negatives = same_artist[: max(4, candidate_size // 2)] + others[: candidate_size - 1]
        negatives = negatives[: candidate_size - 1]
        if len(negatives) < candidate_size - 1:
            continue

        candidate_tracks = negatives + [target_id]
        rng.shuffle(candidate_tracks)
        candidates = []
        for idx, track_id in enumerate(candidate_tracks, start=1):
            meta = track_meta[track_id]
            candidates.append(
                {
                    "candidate_id": f"C{idx:02d}",
                    "track_id": track_id,
                    "artist_name": meta["artist_name"],
                    "track_name": meta["track_name"],
                    "label": track_label(meta["artist_name"], meta["track_name"]),
                }
            )
        examples.append(
            Example(
                dataset="mpd",
                example_id=f"mpd::{playlist['pid']}",
                context_type="playlist_prefix",
                context_title=playlist.get("name"),
                user_id=f"playlist_{playlist['pid']}",
                history=[track_label(track["artist_name"], track["track_name"]) for track in prefix[-10:]],
                history_artists=[track["artist_name"] for track in prefix[-10:]],
                top_artists=top_artists,
                country=None,
                target_id=target_id,
                target_label=track_label(target["artist_name"], target["track_name"]),
                candidates=candidates,
            )
        )
        if len(examples) >= max_examples:
            break

    metadata = {
        "num_playlists_total": len(playlists),
        "num_playlists_train": len(playlists),
        "num_playlists_test": len(examples),
        "candidate_pool_size": len(candidate_pool),
        "track_meta": track_meta,
        "track_popularity": dict(popularity.most_common(20)),
        "train_sequences": train_sequences,
    }
    return examples, metadata


def build_cooccurrence_model(sequences: list[list[str]], window_size: int = 10) -> dict[str, Counter[str]]:
    cooc: dict[str, Counter[str]] = defaultdict(Counter)
    for seq in sequences:
        for idx, item in enumerate(seq):
            left = max(0, idx - window_size)
            right = min(len(seq), idx + window_size + 1)
            for j in range(left, right):
                if j == idx:
                    continue
                other = seq[j]
                cooc[item][other] += 1
    return cooc


def score_popularity(example: Example, popularity: Counter[str]) -> list[str]:
    scored = [(cand["candidate_id"], popularity[cand["track_id"]]) for cand in example.candidates]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return [candidate_id for candidate_id, _ in scored]


def score_cooccurrence(
    example: Example,
    cooc: dict[str, Counter[str]],
    artist_cooc: dict[str, Counter[str]],
    popularity: Counter[str],
    track_to_artist: dict[str, str],
) -> list[str]:
    history_track_ids = [label_to_track_id[label] for label in example.history if label in label_to_track_id]
    recent_weighted = list(enumerate(history_track_ids, start=1))
    scores = []
    for cand in example.candidates:
        score = 0.0
        for idx, history_track_id in recent_weighted:
            weight = idx / len(recent_weighted)
            score += weight * cooc.get(history_track_id, {}).get(cand["track_id"], 0.0)
            history_artist = track_to_artist.get(history_track_id)
            candidate_artist = cand["artist_name"]
            score += 0.6 * weight * artist_cooc.get(history_artist, {}).get(candidate_artist, 0.0)
        if cand["artist_name"] in example.top_artists:
            score += 1.5
        score += math.log1p(popularity[cand["track_id"]]) * 0.05
        scores.append((cand["candidate_id"], score))
    scores.sort(key=lambda pair: (-pair[1], pair[0]))
    return [candidate_id for candidate_id, _ in scores]


def make_prompt(example: Example, prompt_style: str) -> str:
    candidate_lines = "\n".join(
        f"{cand['candidate_id']}: {cand['label']}" for cand in example.candidates
    )
    history_lines = "\n".join(f"- {label}" for label in example.history)
    profile_bits = []
    if example.context_title:
        profile_bits.append(f"Playlist title: {example.context_title}")
    if example.country:
        profile_bits.append(f"Country: {example.country}")
    if example.top_artists:
        profile_bits.append(f"Top artists in context/history: {', '.join(example.top_artists[:5])}")
    profile_text = "\n".join(profile_bits)

    template_name = "minimal_prompt.txt" if prompt_style == "minimal" else "profile_prompt.txt"
    template = Path("prompts") / template_name
    return template.read_text().format(
        context_type=example.context_type,
        history_lines=history_lines,
        candidate_lines=candidate_lines,
        profile_text=profile_text,
    )


def reciprocal_rank(ranked_ids: list[str], positive_id: str) -> float:
    try:
        return 1.0 / (ranked_ids.index(positive_id) + 1)
    except ValueError:
        return 0.0


def hit_at_k(ranked_ids: list[str], positive_id: str, k: int) -> float:
    return float(positive_id in ranked_ids[:k])


def ndcg_at_k(ranked_ids: list[str], positive_id: str, k: int) -> float:
    try:
        rank = ranked_ids.index(positive_id) + 1
    except ValueError:
        return 0.0
    if rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def bootstrap_ci(values: list[float], rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(sample.mean()))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_cohens_d(a: list[float], b: list[float]) -> float:
    diff = np.array(a) - np.array(b)
    if np.std(diff, ddof=1) == 0:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def bh_adjust(p_values: list[float]) -> list[float]:
    p = np.array(p_values, dtype=float)
    order = np.argsort(p)
    ranked = np.empty_like(order, dtype=float)
    n = len(p)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        adjusted = min(prev, p[idx] * n / (i + 1))
        ranked[idx] = adjusted
        prev = adjusted
    return ranked.tolist()


def popularity_percentile(track_id: str, popularity: Counter[str], sorted_items: list[str]) -> float:
    try:
        rank = sorted_items.index(track_id)
    except ValueError:
        return 1.0
    if len(sorted_items) == 1:
        return 0.0
    return rank / (len(sorted_items) - 1)


def evaluate_rankings(
    examples: list[Example],
    method_rankings: dict[str, dict[str, list[str]]],
    popularity: Counter[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    popularity_order = [track_id for track_id, _ in popularity.most_common()]
    rows = []
    top10_by_method: dict[str, list[str]] = defaultdict(list)
    for example in examples:
        positive_candidate_id = next(
            cand["candidate_id"] for cand in example.candidates if cand["track_id"] == example.target_id
        )
        for method, ranking_map in method_rankings.items():
            ranked_ids = ranking_map[example.example_id]
            top10_track_ids = [
                next(cand["track_id"] for cand in example.candidates if cand["candidate_id"] == candidate_id)
                for candidate_id in ranked_ids[:10]
            ]
            top10_by_method[method].extend(top10_track_ids)
            rows.append(
                {
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "method": method,
                    "rr": reciprocal_rank(ranked_ids, positive_candidate_id),
                    "hit10": hit_at_k(ranked_ids, positive_candidate_id, 10),
                    "ndcg10": ndcg_at_k(ranked_ids, positive_candidate_id, 10),
                    "mean_top10_popularity_pct": safe_mean(
                        [popularity_percentile(track_id, popularity, popularity_order) for track_id in top10_track_ids]
                    ),
                    "target_artist_in_profile": float(
                        next(cand for cand in example.candidates if cand["track_id"] == example.target_id)["artist_name"]
                        in example.top_artists
                    ),
                }
            )
    per_example = pd.DataFrame(rows)
    per_example.to_csv(output_dir / "per_example_metrics.csv", index=False)

    rng = np.random.default_rng(42)
    summary_rows = []
    for (dataset, method), frame in per_example.groupby(["dataset", "method"], sort=False):
        for metric in ["rr", "hit10", "ndcg10", "mean_top10_popularity_pct"]:
            values = frame[metric].tolist()
            ci_low, ci_high = bootstrap_ci(values, rng)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": len(values),
                }
            )
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "metric": "coverage",
                "mean": len(set(top10_by_method[method])) / max(1, len(popularity_order)),
                "std": 0.0,
                "min": len(set(top10_by_method[method])) / max(1, len(popularity_order)),
                "max": len(set(top10_by_method[method])) / max(1, len(popularity_order)),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n": len(frame),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    tests = []
    primary_methods = [method for method in per_example["method"].unique() if method != "popularity"]
    raw_p_values = []
    test_index = []
    for dataset in per_example["dataset"].unique():
        dataset_frame = per_example[per_example["dataset"] == dataset]
        pop_frame = dataset_frame[dataset_frame["method"] == "cooccurrence"].set_index("example_id")
        for method in primary_methods:
            if method == "cooccurrence":
                continue
            comp_frame = dataset_frame[dataset_frame["method"] == method].set_index("example_id")
            joined = pop_frame.join(comp_frame, lsuffix="_base", rsuffix="_comp", how="inner")
            for metric in ["rr", "ndcg10"]:
                base = joined[f"{metric}_base"].tolist()
                comp = joined[f"{metric}_comp"].tolist()
                stat, p_value = wilcoxon(comp, base, zero_method="wilcox", alternative="two-sided")
                raw_p_values.append(float(p_value))
                test_index.append((dataset, method, metric))
                tests.append(
                    {
                        "dataset": dataset,
                        "baseline": "cooccurrence",
                        "method": method,
                        "metric": metric,
                        "wilcoxon_stat": float(stat),
                        "p_value": float(p_value),
                        "effect_size_d": paired_cohens_d(comp, base),
                        "method_mean": float(np.mean(comp)),
                        "baseline_mean": float(np.mean(base)),
                    }
                )
    adjusted = bh_adjust(raw_p_values) if raw_p_values else []
    for row, adj in zip(tests, adjusted):
        row["p_value_bh"] = adj
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(output_dir / "significance_tests.csv", index=False)
    return per_example, summary, tests_df


def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plot_metrics = ["rr", "ndcg10", "hit10", "mean_top10_popularity_pct", "coverage"]
    for metric in plot_metrics:
        frame = summary[summary["metric"] == metric].copy()
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(data=frame, x="dataset", y="mean", hue="method")
        ax.set_title(f"{metric} by dataset and method")
        ax.set_ylabel(metric)
        ax.set_xlabel("dataset")
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_comparison.png", dpi=200)
        plt.close()


def save_examples(examples: list[Example], output_path: Path) -> None:
    rows = [asdict(example) for example in examples]
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    env_info = {
        "python": sys.version,
        "seed": args.seed,
        "openai_model": args.openai_model,
        "claude_model": args.claude_model,
        "timestamp": time.time(),
    }

    lastfm_examples, lastfm_meta = build_lastfm_examples(args.lastfm_examples, args.candidate_size, args.seed)
    mpd_examples, mpd_meta = build_mpd_examples(args.mpd_examples, args.candidate_size, args.seed)
    examples = lastfm_examples + mpd_examples
    save_examples(lastfm_examples, results_dir / "lastfm_examples.json")
    save_examples(mpd_examples, results_dir / "mpd_examples.json")

    global label_to_track_id
    label_to_track_id = {}
    for example in examples:
        for cand in example.candidates:
            label_to_track_id[cand["label"]] = cand["track_id"]

    lastfm_popularity = Counter()
    for sequence in lastfm_meta["train_sequences"]:
        lastfm_popularity.update(sequence)
    mpd_popularity = Counter()
    for sequence in mpd_meta["train_sequences"]:
        mpd_popularity.update(sequence)

    lastfm_cooc = build_cooccurrence_model(lastfm_meta["train_sequences"])
    lastfm_artist_cooc = build_cooccurrence_model(lastfm_meta["train_artist_sequences"])
    mpd_cooc = build_cooccurrence_model(mpd_meta["train_sequences"])
    mpd_artist_sequences = [[mpd_meta["track_meta"][track_id]["artist_name"] for track_id in seq] for seq in mpd_meta["train_sequences"]]
    mpd_artist_cooc = build_cooccurrence_model(mpd_artist_sequences)
    lastfm_track_to_artist = {track_id: meta["artist_name"] for track_id, meta in lastfm_meta["track_to_artist"].items()}
    mpd_track_to_artist = {track_id: meta["artist_name"] for track_id, meta in mpd_meta["track_meta"].items()}

    llm_client = LLMClient(
        cache_path=results_dir / "model_outputs" / "response_cache.jsonl",
        openai_model=args.openai_model,
        claude_model=args.claude_model,
    )

    rankings: dict[str, dict[str, list[str]]] = {
        "popularity": {},
        "cooccurrence": {},
        "openai_minimal": {},
        "openai_profile": {},
        "claude_minimal": {},
        "claude_profile": {},
    }

    for example in examples:
        popularity = lastfm_popularity if example.dataset == "lastfm" else mpd_popularity
        cooc = lastfm_cooc if example.dataset == "lastfm" else mpd_cooc
        artist_cooc = lastfm_artist_cooc if example.dataset == "lastfm" else mpd_artist_cooc
        track_to_artist = lastfm_track_to_artist if example.dataset == "lastfm" else mpd_track_to_artist
        rankings["popularity"][example.example_id] = score_popularity(example, popularity)
        rankings["cooccurrence"][example.example_id] = score_cooccurrence(
            example,
            cooc,
            artist_cooc,
            popularity,
            track_to_artist,
        )

        candidate_ids = [cand["candidate_id"] for cand in example.candidates]
        for provider in ["openai", "claude"]:
            for prompt_style in ["minimal", "profile"]:
                method = f"{provider}_{prompt_style}"
                prompt = make_prompt(example, prompt_style)
                rankings[method][example.example_id] = llm_client.rank_candidates(
                    provider=provider,
                    prompt_name=prompt_style,
                    example=example,
                    prompt=prompt,
                    candidate_ids=candidate_ids,
                )

    combined_rows = []
    for dataset_name, dataset_examples, popularity in [
        ("lastfm", lastfm_examples, lastfm_popularity),
        ("mpd", mpd_examples, mpd_popularity),
    ]:
        per_example, summary, tests = evaluate_rankings(
            dataset_examples,
            rankings,
            popularity,
            results_dir / dataset_name,
        )
        plot_results(summary, figures_dir / dataset_name)
        combined_rows.append(summary.assign(dataset_name=dataset_name))

    env_info["gpu_info"] = os.popen(
        "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo NO_GPU"
    ).read()
    env_info["lastfm_meta"] = {
        key: value for key, value in lastfm_meta.items() if key not in {"track_to_artist", "train_sequences", "train_artist_sequences"}
    }
    env_info["mpd_meta"] = {
        key: value for key, value in mpd_meta.items() if key not in {"track_meta", "train_sequences"}
    }
    (results_dir / "config.json").write_text(json.dumps(env_info, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lastfm-examples", type=int, default=60)
    parser.add_argument("--mpd-examples", type=int, default=60)
    parser.add_argument("--candidate-size", type=int, default=20)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--openai-model", default="gpt-4.1")
    parser.add_argument("--claude-model", default="anthropic/claude-sonnet-4.5")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
