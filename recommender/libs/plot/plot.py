from typing import Dict, Any, List, Optional
import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
sns.set_style("darkgrid")

from recommender.libs.constant.inference.recommend import TOP_K_VALUES
from recommender.libs.constant.inference.evaluation import Metric


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
                    "@k": [f"@{k}"]*len(epochs),
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
            "loss": ["training"]*len(tr_loss) + ["validation"]*len(val_loss),
            "epochs": epochs * 2,
        }
    )
    plot_metric(
        df=tr_loss_df,
        metric_name="loss",
        save_path=os.path.join(parent_save_path, "loss.png"),
        hue="loss",
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
    plt.savefig(save_path)
    plt.show()
    plt.close()