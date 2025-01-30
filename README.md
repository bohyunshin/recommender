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

You can select model that you want to train and set appropriate loss function.

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
  --loss mse \
  --epochs 30 \
  --num_factors 16 \
  --train_ratio 0.8 \
  --random_state 42 \
  --result_path "./result/std"
```

### Check experiment result

The results will be saved in `./result/svd` where you can check figures, logs and model weights.

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

## Implemented models

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

Refer to following parameter description when running `recommender/train.py` or `recommender/train_csr.py`.
You can check below parameters in [this code](https://github.com/bohyunshin/recommender/blob/master/recommender/libs/utils/parse_args.py). 

<details><summary>Parameter explanations</summary>

| Parameter name        | Explanation                                                               | default  |
|-----------------------|---------------------------------------------------------------------------|----------|
| `dataset`             | integrated dataset name                                                   | required |
| `model`               | implemented model name                                                    | required |
| `loss`                | implemented loss name                                                     | required |
| `implicit`            | whether implicit dataset type or not                                      | False    |
| `num_neg`             | number of negative samples                                                | None     |
| `neg_sample_strategy` | negative sampling strategy                                                | None     |
| `batch_size`          | number of data in one batch                                               | 128      |
| `lr`                  | learning rate controlling speed of gradient descent                       | 1e-2     |
| `regularization`      | hyper parameter controlling balance between original loss and penalty     | 1e-4     |
| `epochs`              | number of training epochs                                                 | 10       |
| `num_factors`         | dimension of user embedding and item embedding                            | 128      |
| `train_ratio`         | ratio of training dataset                                                 | 0.8      |
| `random_state`        | random seed for reproducibility                                           | 42       |
| `patience`            | tolerance count when validation loss does not drop                        | 5        |
| `result_path`         | absolute directory to store training result                               | required |
| `num_sim_user_top_N`  | number of users who are the most similar in top N (used in user_based CF) | 45       |
| `test`                | when set true, use part of dataset when training for quick pytest         | False    |
</details>

## Supported dataset

For more details about how to download dataset in local, please refer `scripts/download/README.md`.

| Name            | description                                               |
|-----------------|-----------------------------------------------------------|
| `movielens 1m`  | movie rating dataset with user metadata and item metadata |


## Supported loss

You can choose which loss to use when training models. Here is a list of loss that we are currently supporting.

| Name                        | type                      |
|-----------------------------|---------------------------|
| `Mean Squared Loss`         | regression / explicit     |
| `Binary Cross Entropy Loss` | classification / implicit |
| `Triplet Loss (BPR)`        | triplet / implicit        |

Figure out which loss is best suited for specific dataset !


## How to contribute
Although any kinds of PRs are warmly welcomed, please refer to following rules.

* After opening PRs, all the integration tests should be passed.
* Basic tests in `tests/` directory should be added.
* Depending on input data type (pytorch dataset or csr matrix), you should integrate your implementation to `recommender/train.py` or `recommender/train_csr.py`.
* When adding new models, please include followings in `PR`.
  * Experiment results including metric plot, metric value, loss plot after running `recommender/train.py` or `recommender/train_csr.py` with your arguments.
  * Example command to reproduce model training result.
  * Full logs when executing model training python script.

