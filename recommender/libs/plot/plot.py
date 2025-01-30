import os
import pickle
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

sns.set_style("darkgrid")

from recommender.libs.constant.inference.evaluation import Metric
from recommender.libs.constant.inference.recommend import TOP_K_VALUES
from recommender.libs.constant.save.save import FileName


def plot_metric_at_k(
    metric: Dict[int, Dict[str, Any]],
    tr_loss: List[float],
    val_loss: List[float],
    parent_save_path: str,
) -> None:
    """
    Draw metrics line plot @k at each epoch after training.
    For direct recommendation, @3,7,10,20 diners will be used.
    For candidate generation, @100,300,500 diners will be used.
    Number of items to used in metrics are different depending on the purpose.

    Args:
        metric (Dict[int, Dict[str, Any]]): metric object after training.
        tr_loss (List[float]): train loss value in each epoch.
        val_loss (List[float]): validation loss value in each epoch.
        parent_save_path (str): parent save path which will be joined with metric name.
    """
    pred_metrics = [
        Metric.MAP.value,
        Metric.NDCG.value,
        Metric.RECALL.value,
    ]
    epochs = [i for i in range(len(tr_loss))]

    for metric_name in pred_metrics:
        pred_metrics_df = pd.DataFrame()
        for k in TOP_K_VALUES:
            tmp = pd.DataFrame(
                {
                    "@k": [f"@{k}"] * len(epochs),
                    "value": metric[k][metric_name],
                    "epochs": epochs,
                }
            )
            pred_metrics_df = pd.concat([pred_metrics_df, tmp])
        plot_metric(
            df=pred_metrics_df,
            metric_name=metric_name,
            save_path=os.path.join(parent_save_path, f"{metric_name}.png"),
            hue="@k",
        )

    tr_loss_df = pd.DataFrame(
        {
            "value": tr_loss + val_loss,
            "loss_type": ["training"] * len(tr_loss) + ["validation"] * len(val_loss),
            "epochs": epochs * 2,
        }
    )
    plot_metric(
        df=tr_loss_df,
        metric_name="loss",
        save_path=os.path.join(parent_save_path, "loss.png"),
        hue="loss_type",
    )


def plot_metric(
    df: pd.DataFrame,
    metric_name: str,
    save_path: str,
    hue: Optional[str],
) -> None:
    """
    Draw line plot given dataframe.

    Args:
        df (pd.DataFrame): dataframe which contains information about metric.
        metric_name (str): metric name to be plotted.
        save_path (str): path to save line plot.
        hue (str, optional): hue field.
    """
    if hue is not None:
        sns.lineplot(x="epochs", y="value", data=df, hue=hue, marker="o")
        title = f"{metric_name} at every epoch by {hue}"
    else:
        sns.lineplot(x="epochs", y="value", data=df, marker="o")
        title = f"{metric_name} at every epoch"
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    plt.ylabel(metric_name)
    plt.title(title)
    plt.show()
    plt.savefig(save_path)
    plt.close()


def compare_metrics_between_models_at_k(
    result_path: str,
    models: List[str],
    num_epochs: int,
) -> None:
    """
    Plot metrics at every epochs among selected models.

    Args:
        result_path (str): Path where metrics.pkl is saved.
        models (List[str]): List of selected models.
        num_epochs (int): Selected number of epochs while training.
    """
    pred_metrics = [
        Metric.MAP.value,
        Metric.NDCG.value,
        Metric.RECALL.value,
    ]
    metrics_at_k = {}
    dir_name = os.path.join(result_path, "aggregate")
    os.makedirs(dir_name, exist_ok=True)
    for model in models:
        metrics_result_path = os.path.join(result_path, model, FileName.METRIC.value)
        metrics_at_k[model] = pickle.load(open(metrics_result_path, "rb"))

    for k in TOP_K_VALUES:
        for metric in pred_metrics:
            values = []
            model_names = []
            epochs = []
            for model in models:
                metric_values = metrics_at_k[model][k][metric]
                remain_epochs = num_epochs - len(metric_values)
                values += metric_values + [metric_values[-1]] * remain_epochs
                model_names += [model] * num_epochs
                epochs += [i for i in range(num_epochs)]
            tmp = pd.DataFrame(
                {
                    "model": model_names,
                    "epochs": epochs,
                    "value": values,
                }
            )
            plot_metric(
                df=tmp,
                metric_name=metric,
                save_path=os.path.join(dir_name, f"{metric}@{k}.png"),
                hue="model",
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--models", type=str, required=True, nargs="+")
    parser.add_argument("--num_epochs", type=int, required=True)
    args = parser.parse_args()

    compare_metrics_between_models_at_k(
        result_path=args.result_path,
        models=args.models,
        num_epochs=args.num_epochs,
    )
