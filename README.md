# recommender

This repository aims for implementing various machine learning algorithms in recommender system. Any PRs are warmly welcomed!

## Why made this repository?
- There are lots of algorithms in recommender system starting from item-based recommendation, matrix factorization to deep-learning based recommendation.
- To understand better, this repository provides implementation of various recommender algorithms using pytorch or custom learning methods.
- Also, we not only offer implementation code but also pipelines for training recommender models with various implicit / explicit data (movielens, yelp, netflix etc..)
- By comparing metric (ndcg, hit ratio etc..) between different dataset and algorithm, figure out which algorithm is suitable for specific situation.

## Current algorithm implementation list

|Dataset category|Algorithm category|Algorithm|Path|
|----------------|---|---|---|
|explicit|matrix factorization|SVD|`recommender/model/mf/explicit_mf`|
|implicit|weighted matrix factorization|ALS|`recommender/model/mf/implicit_mf`|

## Current dataset pipeline list

|Category|Name|
|----------------|---|
|explicit|movielens|

## Experiment results

TBD