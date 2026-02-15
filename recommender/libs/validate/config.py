from argparse import ArgumentParser

from recommender.libs.constant.data.name import DatasetName
from recommender.libs.constant.loss.name import LossName
from recommender.libs.constant.model.name import ModelName
from recommender.libs.constant.sampling.negative_sampling import (
    NegativeSamplingStrategy,
)


def validate_config(args: ArgumentParser.parse_args):
    if args.loss == LossName.MSE:
        # rating data only exists in movielens
        assert args.dataset in [
            DatasetName.MOVIELENS_1M,
            DatasetName.MOVIELENS_10M,
        ]
        # mse loss function is possible in only svd based models
        assert args.model in [ModelName.SVD, ModelName.SVD_BIAS]
    # als loss is possible only for als
    if args.loss == LossName.ALS:
        assert args.model == ModelName.ALS
    # negative sampling config should be set when related model or loss
    if args.model in [
        ModelName.GMF,
        ModelName.MLP,
        ModelName.TWO_TOWER,
    ] or args.loss in [LossName.BPR, LossName.BCE]:
        assert args.implicit is True
        assert args.num_neg is not None
        assert args.neg_sample_strategy is not None
    # check negative sampling strategy
    for strategy in args.neg_sample_strategy:
        assert strategy in list(NegativeSamplingStrategy)
