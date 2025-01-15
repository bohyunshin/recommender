# recommender
[![codecov](https://codecov.io/github/bohyunshin/recommender/graph/badge.svg?token=SCB83VOII7)](https://codecov.io/github/bohyunshin/recommender)

This repository aims for implementing various machine learning algorithms in recommender system with unified pipeline. Any PRs are warmly welcomed!

## Setting up environment

Ensure that latest `poetry` version is installed.

```shell
$ poetry --version
Poetry (version 1.8.5)
```

Python version higher than 3.11 is required.

```shell
$ python3 --version
Python 3.11.11
```

Make virtual environment using poetry.

```shell
$ poetry shell
```

Install required packages from `poetry.lock`.

```shell
$ poetry install
```

## Quick start

Let's run singular value decomposition on movielens 1m dataset using our pipeline.

### Download dataset

Dataset will be downloaded in `recommender/.data/movielens`.

```bash
$ python3 scripts/download/movielens.py --package ml-1m
```

### Run pipeline
There are two kinds of `main scripts` depending on model type.

- torch based model: `recommender/train.py`
- csr based model: `recommender/train_csr.py`

Because svd is torch based model, run `recommender/train.py`.

```bash
$ python3 recommender/train.py \
  --dataset movielens \
  --model svd \
  --epochs 30 \
  --num_factors 16 \
  --train_ratio 0.8 \
  --random_state 42 \
  --result_path "./result"
```

### Check experiment result

The results will be saved in `./result/svd` where you can check figures, logs and model weights

```bash
$ ls
log.log
map.png
model.pt
recall.png
validation_loss.pkl
loss.png
metric.pkl
ndcg.png
training_loss.pkl
weight.pt
```

## Why made this repository?
- There are lots of algorithms in recommender system starting from item-based recommendation, matrix factorization to deep-learning based recommendation.
- To understand better, this repository provides implementation of various recommender algorithms using pytorch or custom learning methods.
- Also, we not only offer implementation code but also pipelines for training recommender models with various implicit / explicit data (movielens, yelp, pinterest etc..)
- By comparing metric (ndcg, mAP etc..) between different dataset and algorithm, figure out which algorithm is suitable for specific situation.

## Architectures

```mermaid
flowchart LR
    A[Download data] --> B[Load data]
    B --> C[Preprocess data]
    C --> D[Prepare model data]
    D --> E[Training]
    E --> F[Summarize results]
```

|Step|Code|Description|
|----|----|-----------|
|Download data|`scripts/download`|Download selected dataset from public url|
|Load data|`recommender/load_data`|Load downloaded dataset with pandas data type|
|Preprocess data|`recommender/preprocess`|Preprocess dataset|
|Prepare model data|`recommender/prepare_model_data`|Convert dataset which will be fed to models|
|Training|`recommender/model`|Train various recommender algorithms|
|Summarize results|`recommender/libs/plot`|Make metric plots, loss curve|

## Algorithms currently implemented

|Dataset category|Algorithm|Input data type|Path|Loss|
|----------------|---|---|---|---|
|explicit|User based CF|csr matrix|`recommender/model/neighborhood/user_based.py`|NA|
|explicit|SVD|pytorch dataset|`recommender/model/mf/svd.py`|$L = \sum_{(u,i) \in \mathcal{K}} (r_{ui} - p_u^T q_i)^2 $|
|explicit|SVD with bias|pytorch dataset|`recommender/model/mf/svd.py`|$L = \sum_{(u,i) \in \mathcal{K}} (r_{ui} - p_u^T q_i - b_u - b_i - \mu)^2 + \lambda (\| p_u \|^2 + \| q_i \|^2 + \| b_u \|^2 + \| b_i \|^2) $|
|implicit|ALS|csr matrix|`recommender/model/mf/als.py`|$L = \sum_{u,i} c_{ui}(r_{ui} - p_u^T q_i)^2 - \lambda (\| p_u \|^2 + \| q_i \|^2)$|
|implicit|BPR|pytorch dataset|`recommender/model/bpr.py`|$L = \sum_{(u,i,j) \in D_S} \log \ \sigma(\hat{x}_{uij}) - \lambda (\| p_u \|^2 + \| q_i \|^2)$|
|implicit|GMF|pytorch dataset|`recommender/model/deep_learning/gmf.py`|$L = \sum_{u,i} ( b_{ui} \log \  \sigma (h^T (p_u \odot q_i)) + (1-b_{ui}) \log \  ( 1-\sigma (h^T (p_u \odot q_i)) ) )$|
|implicit|MLP|pytorch dataset|`recommender/model/deep_learning/mlp.py`|$L = \sum_{u,i} ( b_{ui} \log \  \sigma (h^T Z(p_u, q_i)) + (1-b_{ui}) \log \  ( 1-\sigma (h^T Z(p_u, q_i)) ) )$|
|implicit|TWO-TOWER|pytorch dataset|`recommender/model/deep_learning/two_tower.py`|$L = \sum_{u,i} ( b_{ui} \log \  \sigma (h^T concat(Z_u(p_u, m_u), Z_i(q_i, m_i)) ) + (1-b_{ui}) \log \  (1-\sigma (h^T concat(Z_u(p_u, m_u), Z_i(q_i, m_i)) ) )  )$|

## How to contribute
Although any kinds of PRs are warmly welcomed, please refer to following rules.

* After opening PRs, all the integration tests should be passed.
* Basic tests in `tests/` directory should be added.
* Depending on input data type (pytorch dataset or csr matrix), you should integrate your implementation to `recommender/train.py` or `recommender/train_csr.py`.
* When adding new models, please include followings in `PR`.
  * Experiment results including metric plot, metric value, loss plot after running `recommender/train.py` or `recommender/train_csr.py` with your arguments.
  * Example command to reproduce model training result.
  * Full logs when executing model training python script.

## Dataset currently supported

For more details about how to download dataset in local, please refer `scripts/download/README.md`.

|Name            |type    |
|----------------|--------|
|movielens 1m    |explicit|

## Experiment results

To reproduce following experiment results, please refer to `Code to reproduce` column in following summary table.

### movielens 1m dataset result

|Dataset category|Algorithm|mAP@10|mAP@20|mAP@50|NDCG@10|NDCG@20|NDCG@50|Code to reproduce|
|----------------|---------|------|------|------|-------|-------|-------|-----------------|
|explicit|SVD|0.0532|0.0385|0.0281|0.1052|0.0959|0.094|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \ &#13;  --dataset movielens \ &#13;  --model svd \ &#13;  --epochs 50 \ &#13;  --num_factors 16 \ &#13;  --train_ratio 0.8 \ &#13;  --random_state 42 \ &#13;  --result_path "./result/svd"</pre></details>|
|explicit|SVD with bias|0.0245|0.0185|0.015|0.0598|0.0585|0.0661|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \&#13;  --dataset movielens \&#13;  --model svd_bias \&#13;  --epochs 50 \&#13;  --num_factors 16 \&#13;  --train_ratio 0.8 \&#13;  --random_state 42 \&#13;  --result_path "./result/svd_bias"</pre></details>|
|implicit|ALS|0.2509|0.2083|0.1856|0.3789|0.3616|0.3783|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train_csr.py \&#13;  --dataset movielens \&#13;  --model als \&#13;  --implicit \&#13;  --epochs 50 \&#13;  --num_factors 16 \&#13;  --train_ratio 0.8 \&#13;  --random_state 42 \&#13;  --result_path "./result/als"</pre></details>|
|implicit|BPR|0.2163|0.1737|0.1457|0.3265|0.3054|0.3131|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \&#13;  --dataset movielens \&#13;  --model bpr \&#13;  --implicit \&#13;  --epochs 50 \&#13;  --num_factors 16 \&#13;  --train_ratio 0.8 \&#13;  --random_state 42 \&#13;  --result_path "./result/bpr"</pre></details>|
|implicit|GMF|0.003|0.0021|0.0017|0.0094|0.0099|0.0126|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \&#13;  --dataset movielens \&#13;  --model gmf \&#13;  --implicit \&#13;  --epochs 50 \&#13;  --num_factors 16 \&#13;  --train_ratio 0.8 \&#13;  --random_state 42 \&#13;  --result_path "./result/gmf"</pre></details>|
|implicit|MLP|0.1639|0.1303|0.11|0.2673|0.253|0.2643|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \&#13;  --dataset movielens \&#13;  --model mlp \&#13;  --implicit \&#13;  --epochs 50 \&#13;  --num_factors 16 \&#13;  --train_ratio 0.8 \&#13;  --random_state 42 \&#13;  --result_path "./result/mlp"</pre></details>|
|implicit|TWO-TOWER|0.118|0.0908|0.072|0.2039|0.1916|0.1933|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \&#13;  --dataset movielens \&#13;  --model two_tower \&#13;  --implicit \&#13;  --epochs 50 \&#13;  --num_factors 16 \&#13;  --train_ratio 0.8 \&#13;  --random_state 42 \&#13;  --result_path "./result/two_tower"</pre></details>|

The results that ALS has highest performance on test dataset are pretty interesting, indicating that neural network is not always the best solution.

### movielens 10m dataset result
TBD

### Note
* Experiment of user-based CF using movielens are not performed yet because of excessive time required.
* Implementation of cython (or etc..) is required to improve training time.

