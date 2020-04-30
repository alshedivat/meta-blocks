"""Integration test for training and evaluation on Omniglot."""

import glob
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile

import pytest
from hydra.experimental import compose as hydra_compose
from hydra.experimental import initialize as hydra_init

from meta_blocks.experiment.eval import evaluate
from meta_blocks.experiment.train import train

logger = logging.getLogger(__name__)

AVAILABLE_METHODS = ("maml", "fomaml", "reptile", "proto")
AVAILABLE_SETTINGS = ("classic_supervised", "self_supervised")
OMNIGLOT_URL = "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/"

# Initialize hydra.
hydra_init(config_dir="conf", strict=False)


@pytest.mark.parametrize("adaptation_method", AVAILABLE_METHODS)
@pytest.mark.parametrize("experiment_setting", AVAILABLE_SETTINGS)
def test_omniglot_integration(adaptation_method, experiment_setting):
    def fetch_omniglot(dir_path):
        omniglot_dir = os.path.join(dir_path, "omniglot")
        os.makedirs(omniglot_dir, exist_ok=False)
        for suffix in ["small1", "small2"]:
            name = f"images_background_{suffix}.zip"
            url = OMNIGLOT_URL + name
            tmp_path = os.path.join(dir_path, "tmp.zip")
            urllib.request.urlretrieve(url, tmp_path)
            with zipfile.ZipFile(tmp_path, "r") as zip_ref:
                zip_ref.extractall(dir_path)
            extracted_dir = os.path.join(dir_path, name.split(".")[0])
            for category_dir in glob.glob(os.path.join(extracted_dir, "*")):
                category_name = os.path.basename(category_dir)
                shutil.move(category_dir, os.path.join(omniglot_dir, category_name))

    with tempfile.TemporaryDirectory() as dir_path:
        # Fetch Omniglot.
        logger.info(f"Fetching omniglot...")
        fetch_omniglot(dir_path)
        data_path = os.path.join(dir_path, "omniglot")
        cfg = hydra_compose(
            config_file="config.yaml",
            overrides=[
                f"adaptation={adaptation_method}",
                f"test=omniglot/{experiment_setting}",
                f"data=omniglot",
                f"network=omniglot",
                f"data.source.data_dir={data_path}",
            ],
            strict=False,
        )
        # Run train and eval.
        logger.info(f"Training...")
        if cfg.train is not None:
            train(cfg, work_dir=dir_path)
        logger.info(f"Evaluating...")
        if cfg.eval is not None:
            evaluate(cfg, work_dir=dir_path)
