from argparse import ArgumentParser

from recommender.libs.constant.loss.name import LossName
from recommender.libs.constant.data.name import DatasetName
from recommender.libs.constant.model.name import ModelName
from recommender.libs.constant.sampling.negative_sampling import NegativeSamplingStrategy


def validate_config(args: ArgumentParser.parse_args):
    # rating data only exists in movielens
    if args.loss == LossName.MSE.value:
        assert args.dataset == DatasetName.MOVIELENS.value
    # negative sampling config should be set when related model or loss
    if args.model in [ModelName.GMF.value, ModelName.MLP.value, ModelName.TWO_TOWER.value] or \
            args.loss in [LossName.BPR.value, LossName.BCE.value]:
        assert args.implicit is True
        assert args.num_neg is not None
        assert args.neg_sample_strategy is not None
    # check negative sampling strategy
    for strategy in args.neg_sample_strategy:
        assert strategy in [enum.value for enum in NegativeSamplingStrategy]