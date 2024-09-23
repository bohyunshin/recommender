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

## Current dataset pipeline list

|Category|Name|
|----------------|---|
|explicit|movielens|

## Experiment results

To reproduce following experiment results, please refer to README.md of each dataset.

|Dataset category|Dataset|Algorithm|mAP@10|mAP@20|mAP@50|NDCG@10|NDCG@20|NDCG@50|
|----------------|-------|---------|------|------|------|-------|-------|-------|
|implicit|movielens 1m|ALS|TBD|TBD|TBD|TBD|TBD|TBD|
|implicit|movielens 10m|ALS|TBD|TBD|TBD|TBD|TBD|<pre lang="bash">python3 recommender/train/mf/implicit_mf.py --dataset movielens \ &#13;  --epochs 100</pre>|



<pre lang="json">{&#10;  "id": 10,&#13;  "username": "chucknorris"&#10;}</pre>

<table>
<tr>
<td>
   Dataset category
</td>
<td>
   <pre lang="csharp">
   const int x = 3;
   const string y = "foo";
   readonly Object obj = getObject();
   </pre>
</td>
<td>
  <pre lang="nemerle">
  def x : int = 3;
  def y : string = "foo";
  def obj : Object = getObject();
  </pre>
</td>
<td>
  Variables defined with <code>def</code> cannot be changed once defined. This is similar to <code>readonly</code> or <code>const</code> in C# or <code>final</code> in Java. Most variables in Nemerle aren't explicitly typed like this.
</td>

<tr>
<td>
   <pre lang="csharp">
   const int x = 3;
   const string y = "foo";
   readonly Object obj = getObject();
   </pre>
</td>
<td>
  <pre lang="nemerle">
  def x : int = 3;
  def y : string = "foo";
  def obj : Object = getObject();
  </pre>
</td>
<td>
  Variables defined with <code>def</code> cannot be changed once defined. This is similar to <code>readonly</code> or <code>const</code> in C# or <code>final</code> in Java. Most variables in Nemerle aren't explicitly typed like this.
</td>

</tr>