import argparse

from recommender.libs.constant.data.name import INTEGRATED_DATASET
from recommender.libs.constant.loss.name import IMPLEMENTED_LOSS
from recommender.libs.constant.model.name import IMPLEMENTED_MODELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=INTEGRATED_DATASET
    )
    parser.add_argument("--model", type=str, required=True, choices=IMPLEMENTED_MODELS)
    parser.add_argument("--loss", type=str, required=True, choices=IMPLEMENTED_LOSS)

    # negative sample config
    parser.add_argument("--implicit", action="store_true")
    parser.add_argument("--num_neg", type=int, default=None)
    parser.add_argument("--neg_sample_strategy", nargs="+", default=None)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_factors", type=int, default=128)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--num_sim_user_top_N", type=int, default=45)
    parser.add_argument("--is_test", action="store_true")
    return parser.parse_args()
