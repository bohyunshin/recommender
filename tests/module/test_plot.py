import os
import pickle
from collections import defaultdict

from recommender.libs.plot.plot import compare_metrics_between_models_at_k
from recommender.libs.constant.inference.evaluation import Metric
from recommender.libs.constant.inference.recommend import TOP_K_VALUES
from recommender.libs.constant.model.name import ModelName
from recommender.libs.constant.save.save import FileName


def test_compare_metrics_between_models_at_k():
    models = [
        ModelName.SVD.value,
        ModelName.MLP.value,
    ]
    pred_metrics = [
        Metric.MAP.value,
        Metric.NDCG.value,
        Metric.RECALL.value,
    ]
    diff = 0
    num_epochs = 5
    dir_name = os.path.join(os.path.dirname(__file__), "result")
    for model in models:
        os.makedirs(os.path.join(dir_name, model), exist_ok=True)
        metrics_at_k = defaultdict(dict)
        for k in TOP_K_VALUES:
            for metric in pred_metrics:
                metrics_at_k[k][metric] = [i+diff for i in range(num_epochs)]
        pickle.dump(metrics_at_k, open(os.path.join(dir_name, model, FileName.METRIC.value), "wb"))
        diff += 1
        num_epochs += 5
    compare_metrics_between_models_at_k(
        result_path=dir_name,
        models=models,
        num_epochs=num_epochs
    )