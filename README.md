# recommender

This repository aims for implementing various machine learning algorithms in recommender system with unified pipeline. Any PRs are warmly welcomed!

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

|Dataset category|Algorithm category|Algorithm|Input data type|Path|Loss|
|----------------|---|---|---|---|---|
|explicit|matrix factorization|SVD|pytorch dataset|`recommender/model/mf/explicit_mf.py`|$L = \sum_{(u,i) \in \mathcal{K}} (r_{ui} - p_u^T q_i)^2 $|
|implicit|weighted matrix factorization|ALS|csr matrix|`recommender/model/mf/implicit_mf.py`|$L = \sum_{u,i} c_{ui}(r_{ui} - p_u^T q_i)^2 - \lambda (\| \| p_u \| \|^2 + \|\|q_i\|\|^2)$|
|implicit|bayesian personalized loss|BPR|pytorch dataset|`recommender/model/bpr.py`|$L = \sum_{(u,i,j) \in D_S} \log \sigmoid(\hat{x}_{uij}) - \lambda (\left p_u\|\|^2 + \|\|q_i\|\|^2)$|

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

|Dataset category|Dataset|Algorithm|mAP@10|mAP@20|mAP@50|NDCG@10|NDCG@20|NDCG@50|Code to reproduce|
|----------------|-------|---------|------|------|------|-------|-------|-------|-----------------|
|explicit|movielens 1m|explicit_mf|TBD|TBD|TBD|TBD|TBD|TBD|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \ &#13;  --dataset movielens \ &#13;  --model explicit_mf \ &#13;  --epochs 30 \ &#13;  --num_factors 16 \ &#13;  --train_ratio 0.8 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-1m \ &#13;  --model_path "../explicit_mf_ml_1m.pkl" \ &#13;  --log_path "../explicit_mf_ml_1m.log" </pre></details>|
|explicit|movielens 10m|explicit_mf|TBD|TBD|TBD|TBD|TBD|TBD|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \ &#13;  --dataset movielens \ &#13;  --model explicit_mf \ &#13;  --epochs 30 \ &#13;  --num_factors 16 \ &#13;  --train_ratio 0.8 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-10m \ &#13;  --model_path "../explicit_mf_ml_10m.pkl" \ &#13;  --log_path "../explicit_mf_ml_10m.log" </pre></details>|
|implicit|movielens 1m|ALS|TBD|TBD|TBD|TBD|TBD|TBD|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train/mf/implicit_mf.py \ &#13;  --dataset movielens \ &#13;  --epochs 30 \ &#13;  --num_factors 10 \ &#13;  --test_ratio 0.2 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-1m \ &#13;  --model_path "../als_ml_1m.pkl" \ &#13;  --log_path "../als_ml_1m.log" </pre></details>|
|implicit|movielens 10m|ALS|TBD|TBD|TBD|TBD|TBD|TBD|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train/mf/implicit_mf.py \ &#13;  --dataset movielens \ &#13;  --epochs 30 \ &#13;  --num_factors 10 \ &#13;  --test_ratio 0.2 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-10m \ &#13;  --model_path "../als_ml_10m.pkl" \ &#13;  --log_path "../als_ml_10m.log" </pre></details>|
|implicit|movielens 1m|BPR|TBD|TBD|TBD|TBD|TBD|TBD|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \ &#13;  --dataset movielens \ &#13;  --model bpr \ &#13;  --implicit \ &#13;  --epochs 30 \ &#13;  --num_factors 16 \ &#13;  --train_ratio 0.8 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-1m \ &#13;  --model_path "../bpr_ml_1m.pkl" \ &#13;  --log_path "../bpr_ml_1m.log" </pre></details>|
|implicit|movielens 10m|BPR|TBD|TBD|TBD|TBD|TBD|TBD|<details><summary>cmd</summary><pre lang="bash">python3 recommender/train.py \ &#13;  --dataset movielens \ &#13;  --model bpr \ &#13;  --implicit \ &#13;  --epochs 30 \ &#13;  --num_factors 16 \ &#13;  --train_ratio 0.8 \ &#13;  --random_state 42 \ &#13;  --movielens_data_type ml-10m \ &#13;  --model_path "../bpr_ml_10m.pkl" \ &#13;  --log_path "../bpr_ml_10m.log" </pre></details>|