import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["movielens", "sample"])
    parser.add_argument("--model", type=str, required=True, choices=["svd", "svd_bias", "als", "bpr", "user_based", "gmf", "mlp", "two_tower"])
    parser.add_argument("--implicit", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_factors", type=int, default=128)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--num_sim_user_top_N", type=int, default=45)
    parser.add_argument("--movielens_data_type", type=str, default="ml-1m")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()