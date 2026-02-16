import importlib
import logging
import os
import pickle
import traceback
from abc import ABC, abstractmethod
from argparse import ArgumentParser

from recommender.libs.constant.data.name import Field
from recommender.libs.constant.save.save import FileName
from recommender.libs.plot.plot import plot_metric_at_k
from recommender.libs.utils.logger import setup_logger


class BaseTrainPipeline(ABC):
    def run(self, args: ArgumentParser.parse_args):
        self._setup(args)
        try:
            self._log_params(args)
            data = self._load_data(args)
            preprocessed_data = self._preprocess(args, data)
            self.num_users = preprocessed_data.get(Field.NUM_USERS)
            self.num_items = preprocessed_data.get(Field.NUM_ITEMS)
            self._prepare_model_data(args, preprocessed_data)
            self._setup_model(args, preprocessed_data)
            self._train(args)
            self._save_artifacts(args)
        except Exception:
            logging.error(traceback.format_exc())
            raise

    def _setup(self, args: ArgumentParser.parse_args):
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        test_dir = "test" if args.is_test else "untest"
        args.result_path = os.path.join("result", test_dir, args.model, timestamp)
        os.makedirs(args.result_path, exist_ok=True)
        setup_logger(os.path.join(args.result_path, FileName.LOG))

    def _load_data(self, args: ArgumentParser.parse_args) -> dict:
        load_data_module = importlib.import_module(
            f"recommender.load_data.{args.dataset}"
        ).LoadData
        return load_data_module().load(is_test=args.is_test)

    def _preprocess(self, args: ArgumentParser.parse_args, data: dict) -> dict:
        preprocess_module = importlib.import_module(
            f"recommender.preprocess.{args.dataset}"
        ).Preprocessor
        return preprocess_module().preprocess(data)

    def _save_metrics_and_losses(self, model, result_path: str):
        pickle.dump(
            model.metric_at_k_total_epochs,
            open(os.path.join(result_path, FileName.METRIC), "wb"),
        )
        pickle.dump(
            model.tr_loss,
            open(os.path.join(result_path, FileName.TRAINING_LOSS), "wb"),
        )
        pickle.dump(
            model.val_loss,
            open(os.path.join(result_path, FileName.VALIDATION_LOSS), "wb"),
        )
        plot_metric_at_k(
            metric=model.metric_at_k_total_epochs,
            tr_loss=model.tr_loss,
            val_loss=model.val_loss,
            parent_save_path=result_path,
        )

    @abstractmethod
    def _log_params(self, args: ArgumentParser.parse_args):
        raise NotImplementedError

    @abstractmethod
    def _prepare_model_data(
        self, args: ArgumentParser.parse_args, preprocessed_data: dict
    ):
        raise NotImplementedError

    @abstractmethod
    def _setup_model(self, args: ArgumentParser.parse_args, preprocessed_data: dict):
        raise NotImplementedError

    @abstractmethod
    def _train(self, args: ArgumentParser.parse_args):
        raise NotImplementedError

    @abstractmethod
    def _save_artifacts(self, args: ArgumentParser.parse_args):
        raise NotImplementedError
