# recommender

This repository aims for implementing various machine learning algorithms in recommender system. Any PRs are warmly welcomed!

## Why made this repository?
- There are lots of algorithms in recommender system starting from item-based recommendation, matrix factorization to deep-learning based recommendation.
- This repository will help you understand various recommender algorithms by implementing them by yourselves.
- Also, we not only offer implementation code but also pipelines for learning recommender models with various implicit / explicit data (movielens, yelp, netflix etc..)
- By comparing metric (ndcg, precision etc..) between different dataset, different algorithm, figure out which algorithm is suitable for specific situation.

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