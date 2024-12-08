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

## Why made this repository?
- There are lots of algorithms in recommender system starting from item-based recommendation, matrix factorization to deep-learning based recommendation.
- To understand better, this repository provides implementation of various recommender algorithms using pytorch or custom learning methods.
- Also, we not only offer implementation code but also pipelines for training recommender models with various implicit / explicit data (movielens, yelp, pinterest etc..)
- By comparing metric (ndcg, mAP etc..) between different dataset and algorithm, figure out which algorithm is suitable for specific situation.

## Architectures
This repository offers code to preprocess raw data, learn models and evaluate trained models.

* [preprocess code](https://github.com/bohyunshin/recommender/tree/master/recommender/preprocess)
* [data loader preparation code](https://github.com/bohyunshin/recommender/tree/master/recommender/data_loader)
* [model implementation code](https://github.com/bohyunshin/recommender/tree/master/recommender/model)
* [evaluation code](https://github.com/bohyunshin/recommender/blob/master/recommender/tools/evaluation.py)

Model training pipelines are unified as a python script, i.e. `recommender/train.py` or `recommender/train_csr.py`, depending on the type of input data.

Therefore, if you want to run model, take a look at the input data type and run the corresponding python script with appropriate arguments.

## Current algorithm implementation list

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

* Basic tests in `tests/` directory should be added.
* Depending on input data type (pytorch dataset or csr matrix), you should integrate your implementation to `recommender/train.py` or `recommender/train_csr.py`.
* When adding new models, please attach followings in `PR` and `README.md`
  * Experiment results using `recommender/train.py` or `recommender/train_csr.py` with your arguments.
  * Example command to reproduce model training result should be added.
  * Logging file when executing model training python script.
  * Training / validation loss for each epoch.

## Current dataset pipeline list

|Category|Name|
|----------------|---|
|explicit/implicit|movielens|

## Experiment results

To reproduce following experiment results, please refer to `Code to reproduce` column in following summary table.

### movielens 1m dataset result

|Dataset category|Algorithm|mAP@10|mAP@20|mAP@50|NDCG@10|NDCG@20|NDCG@50|Code to reproduce|
|----------------|---------|------|------|------|-------|-------|-------|-----------------|
|explicit|SVD|0.0532|0.0385|0.0281|0.1052|0.0959|0.094|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \ &#13;  --dataset movielens \ &#13;  --model svd \ &#13;  --epochs 30 \ &#13;  --num_factors 16 \ &#13;  --train_ratio 0.8 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-1m \ &#13;  --model_path "../svd_ml_1m.pkl" \ &#13;  --log_path "../svd_ml_1m.log" </pre></details>|
|explicit|SVD with bias|0.0245|0.0185|0.015|0.0598|0.0585|0.0661|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \ &#13;  --dataset movielens \ &#13;  --model svd_bias \ &#13;  --epochs 30 \ &#13;  --num_factors 16 \ &#13;  --train_ratio 0.8 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-1m \ &#13;  --model_path "../svd_bias_ml_1m.pkl" \ &#13;  --log_path "../svd_bias_ml_1m.log" </pre></details>|
|implicit|ALS|0.2509|0.2083|0.1856|0.3789|0.3616|0.3783|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train_csr.py \ &#13;  --dataset movielens \ &#13;  --model als \ &#13;  --implicit \ &#13;  --epochs 30 \ &#13;  --num_factors 16 \ &#13;  --train_ratio 0.8 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-1m \ &#13;  --model_path "../als_ml_1m.pkl" \ &#13;  --log_path "../als_ml_1m.log" </pre></details>|
|implicit|BPR|0.2163|0.1737|0.1457|0.3265|0.3054|0.3131|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \ &#13;  --dataset movielens \ &#13;  --model bpr \ &#13;  --implicit \ &#13;  --epochs 30 \ &#13;  --num_factors 16 \ &#13;  --train_ratio 0.8 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-1m \ &#13;  --model_path "../bpr_ml_1m.pkl" \ &#13;  --log_path "../bpr_ml_1m.log" </pre></details>|
|implicit|GMF|0.003|0.0021|0.0017|0.0094|0.0099|0.0126|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \ &#13;  --dataset movielens \ &#13;  --model gmf \ &#13;  --implicit \ &#13;  --epochs 30 \ &#13;  --num_factors 16 \ &#13;  --train_ratio 0.8 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-1m \ &#13;  --model_path "../gmf_ml_1m.pkl" \ &#13;  --log_path "../gmf_ml_1m.log" </pre></details>|
|implicit|MLP|0.1639|0.1303|0.11|0.2673|0.253|0.2643|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \ &#13;  --dataset movielens \ &#13;  --model mlp \ &#13;  --implicit \ &#13;  --epochs 30 \ &#13;  --num_factors 16 \ &#13;  --train_ratio 0.8 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-1m \ &#13;  --model_path "../mlp_ml_1m.pkl" \ &#13;  --log_path "../mlp_ml_1m.log" </pre></details>|
|implicit|TWO-TOWER|0.118|0.0908|0.072|0.2039|0.1916|0.1933|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \ &#13;  --dataset movielens \ &#13;  --model two_tower \ &#13;  --implicit \ &#13;  --epochs 30 \ &#13;  --num_factors 16 \ &#13;  --train_ratio 0.8 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-1m \ &#13;  --model_path "../two_tower_ml_1m.pkl" \ &#13;  --log_path "../two_tower_ml_1m.log" </pre></details>|

The results that ALS has highest performance on test dataset are pretty interesting, indicating that neural network is not always the best solution.

### movielens 10m dataset result
TBD

### Note
* Experiment of user-based CF using movielens are not performed yet because of excessive time required.
* Implementation of cython (or etc..) is required to improve training time.

