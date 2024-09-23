# recommender

This repository aims for implementing various machine learning algorithms in recommender system. Any PRs are warmly welcomed!

## Why made this repository?
- There are lots of algorithms in recommender system starting from item-based recommendation, matrix factorization to deep-learning based recommendation.
- To understand better, this repository provides implementation of various recommender algorithms using pytorch or custom learning methods.
- Also, we not only offer implementation code but also pipelines for training recommender models with various implicit / explicit data (movielens, yelp, netflix etc..)
- By comparing metric (ndcg, mAP etc..) between different dataset and algorithm, figure out which algorithm is suitable for specific situation.

## Current algorithm implementation list

|Dataset category|Algorithm category|Algorithm|Path|
|----------------|---|---|---|
|explicit|matrix factorization|SVD|`recommender/model/mf/explicit_mf`|
|implicit|weighted matrix factorization|ALS|`recommender/model/mf/implicit_mf`|

## How to contribute
Although any kinds of PRs are warmly welcomed, please refer to following rules.

* Basic tests in `tests/` directory should be added.
* When adding new models, please attach following in `PR` and `README.md`
  * Experiment results using current data pipeline.
  * Example command to learn model should be added.
  * Logging file when executing model training python script.
  * Training / validation loss curve.

## Current dataset pipeline list

|Category|Name|
|----------------|---|
|explicit/implicit|movielens|

## Experiment results

To reproduce following experiment results, please refer to `Code to reproduce` column in following summary table.

|Dataset category|Dataset|Algorithm|mAP@10|mAP@20|mAP@50|NDCG@10|NDCG@20|NDCG@50|Code to reproduce|
|----------------|-------|---------|------|------|------|-------|-------|-------|-----------------|
|implicit|movielens 1m|ALS|TBD|TBD|TBD|TBD|TBD|TBD|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train/mf/implicit_mf.py \ &#13;  --dataset movielens \ &#13;  --epochs 100 \ &#13;  --num_factors 10 \ &#13;  --test_ratio 0.2 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-1m \ &#13;  --model_path "../als_ml_1m.pkl" \ &#13;  --log_path "../als_ml_1m.log" </pre></details>|
|implicit|movielens 10m|ALS|TBD|TBD|TBD|TBD|TBD|TBD|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train/mf/implicit_mf.py \ &#13;  --dataset movielens \ &#13;  --epochs 100 \ &#13;  --num_factors 10 \ &#13;  --test_ratio 0.2 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-10m \ &#13;  --model_path "../als_ml_10m.pkl" \ &#13;  --log_path "../als_ml_10m.log" </pre></details>|
