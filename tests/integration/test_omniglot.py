"""Integration test for training and evaluation on Omniglot."""

import glob
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile

import pytest
from hydra._internal.hydra import GlobalHydra, Hydra

from meta_blocks.experiment.eval import evaluate
from meta_blocks.experiment.train import train

logger = logging.getLogger(__name__)

AVAILABLE_METHODS = {"maml", "fomaml", "reptile", "proto"}
OMNIGLOT_URL = "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/"


def get_hydra():
    global_hydra = GlobalHydra()
    if not global_hydra.is_initialized():
        return Hydra.create_main_hydra_file_or_module(
            calling_file=__file__,
            calling_module=None,
            config_dir="configs",
            strict=False,
        )
    else:
        return global_hydra.hydra


@pytest.mark.parametrize("adaptation_method", AVAILABLE_METHODS)
def test_omniglot_integration(adaptation_method):
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
        # Parse hydra configs.
        hydra = get_hydra()
        cfg = hydra.compose_config(
            "config.yaml",
            overrides=[
                f"adaptation={adaptation_method}",
                f"dataset=omniglot",
                f"test=omniglot",
                f"data.read_config.data_dir={data_path}",
            ],
            strict=False,
        )
        # Run train and eval.
        logger.info(f"Training...")
        if cfg.train is not None:
            train(cfg)
        logger.info(f"Evaluating...")
        if cfg.eval is not None:
            evaluate(cfg)
