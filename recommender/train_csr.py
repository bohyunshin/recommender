import os
from argparse import ArgumentParser

from recommender.libs.utils.parse_args import parse_args
from recommender.pipeline.csr import CsrTrainPipeline

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main(args: ArgumentParser.parse_args):
    pipeline = CsrTrainPipeline()
    pipeline.run(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
