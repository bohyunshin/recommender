# How to download dataset

You can download public experiment dataset supported in our repository here.
All dataset will be download in `./recommender/.data/{dataset}` which are specified [in this code](https://github.com/bohyunshin/recommender/tree/master/recommender/libs/constant/data).

## Supported dataset

|Name            |user meta|item meta|type    |download command|
|----------------|---------|---------|--------|----------------|
|movielens 1m    |O        |O        |explicit| <details><summary>cmd</summary><pre lang="bash">python3 scripts/download/movielens.py \ &#13;  --package ml-1m </pre></details>|