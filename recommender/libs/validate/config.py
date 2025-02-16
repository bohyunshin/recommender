from argparse import ArgumentParser

from recommender.libs.constant.data.name import DatasetName
from recommender.libs.constant.loss.name import LossName
from recommender.libs.constant.model.name import ModelName
from recommender.libs.constant.sampling.negative_sampling import (
    NegativeSamplingStrategy,
)


def validate_config(args: ArgumentParser.parse_args):
    if args.loss == LossName.MSE.value:
        # rating data only exists in movielens
        assert args.dataset in [
            DatasetName.MOVIELENS_1M.value,
            DatasetName.MOVIELENS_10M.value,
        ]
        # mse loss function is possible in only svd based models
        assert args.model in [ModelName.SVD.value, ModelName.SVD_BIAS.value]
    # als loss is possible only for als
    if args.loss == LossName.ALS.value:
        assert args.model == ModelName.ALS.value
    # negative sampling config should be set when related model or loss
    if args.model in [
        ModelName.GMF.value,
        ModelName.MLP.value,
        ModelName.TWO_TOWER.value,
    ] or args.loss in [LossName.BPR.value, LossName.BCE.value]:
        assert args.implicit is True
        assert args.num_neg is not None
        assert args.neg_sample_strategy is not None
    # check negative sampling strategy
    for strategy in args.neg_sample_strategy:
        assert strategy in [enum.value for enum in NegativeSamplingStrategy]
