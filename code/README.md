# Cloned Repositories

## A-LLMRec

- URL: https://github.com/ghdtjr/A-LLMRec
- Purpose: KDD 2024 implementation of collaborative-filtering-aligned LLM recommendation
- Location: `code/A-LLMRec/`
- Key files:
  - `main.py`
  - `eval.py`
  - `models/a_llmrec_model.py`
  - `pre_train/sasrec/main.py`
- Notes: uses Amazon review data in the public repo, but the architecture is a strong template for combining a classical recommender with an LLM.

## LLaRA

- URL: https://github.com/ljy0ustc/LLaRA
- Purpose: SIGIR 2024 hybrid sequential recommender with projector-aligned LLM inputs
- Location: `code/LLaRA/`
- Key files:
  - `main.py`
  - `data/lastfm_data.py`
  - `train_lastfm.sh`
  - `recommender/A_SASRec_final_bce_llm.py`
- Notes: directly relevant because the repo includes a LastFM path and reported LastFM results.

## RecBole

- URL: https://github.com/RUCAIBox/RecBole
- Purpose: broad recommendation benchmark framework with many classical and neural baselines
- Location: `code/RecBole/`
- Key files:
  - `run_recbole.py`
  - `recbole/evaluator/metrics.py`
  - `asset/dataset_list.json`
- Notes: strongest reusable baseline framework gathered here; useful for BPR, SASRec, GRU4Rec, item-KNN, and evaluation metrics.

## recsys-smpd

- URL: https://github.com/anaezquerro/recsys-smpd
- Purpose: Spotify MPD challenge baselines for popularity, neighborhood, PureSVD, and track2vec
- Location: `code/recsys-smpd/`
- Key files:
  - `main.py`
  - `split-test.py`
  - `models/baseline.py`
  - `models/puresvd.py`
  - `models/track2vec.py`
  - `utils/evaluation.py`
- Notes: directly useful for recreating classical Spotify playlist-continuation baselines.
