"""Integration test for training and evaluation on mini-ImageNet."""

import logging
import os
import tempfile

import numpy as np
import pytest
from hydra._internal.hydra import GlobalHydra
from hydra.experimental import compose as hydra_compose
from hydra.experimental import initialize as hydra_init
from PIL import Image

from meta_blocks.experiment.eval import evaluate
from meta_blocks.experiment.train import train

logger = logging.getLogger(__name__)

AVAILABLE_METHODS = ("maml", "fomaml", "reptile", "proto")
AVAILABLE_SETTINGS = ("classic_supervised", "self_supervised")

# Initialize hydra.
# TODO: is there a way to check if hydra is initialized using public API?
if not GlobalHydra().is_initialized():
    hydra_init(config_dir="conf", strict=False)


@pytest.mark.parametrize("adaptation_method", AVAILABLE_METHODS)
@pytest.mark.parametrize("experiment_setting", AVAILABLE_SETTINGS)
def test_omniglot_integration(adaptation_method, experiment_setting):
    def generate_dummy_miniimagenet_data(dir_path):
        """Generates dummy data that imitates mini-ImageNet structure.

        Mini-ImageNet is too heavy for integration testing, so we generate
        synthetic data (dummy images) that satisfy the mini-ImageNet spec.
        """
        num_dummy_categories = 20
        num_dummy_img_per_category = 10
        min_img_size, max_img_size = 100, 300
        data_path = os.path.join(dir_path, "miniimagent")
        for set_name in ["train", "valid", "test"]:
            set_path = os.path.join(data_path, set_name)
            for cid in range(num_dummy_categories):
                dummy_category_name = f"n{cid:05d}"
                dummy_category_path = os.path.join(set_path, dummy_category_name)
                os.makedirs(dummy_category_path)
                for img_id in range(num_dummy_img_per_category):
                    img_height = np.random.randint(min_img_size, max_img_size)
                    img_width = np.random.randint(min_img_size, max_img_size)
                    img_array = np.full(
                        (img_height, img_width), img_id * 20, dtype=np.int8
                    )
                    img_path = os.path.join(dummy_category_path, f"{img_id}.JPEG")
                    Image.fromarray(img_array).convert("RGB").save(img_path)

    with tempfile.TemporaryDirectory() as dir_path:
        logger.info(f"Generating dummy mini-ImageNet...")
        generate_dummy_miniimagenet_data(dir_path)
        data_path = os.path.join(dir_path, "miniimagent")
        cfg = hydra_compose(
            config_file="config.yaml",
            overrides=[
                f"adaptation={adaptation_method}",
                f"test=miniimagenet/{experiment_setting}",
                f"data=miniimagenet",
                f"network=miniimagenet",
                f"data.source.data_dir={data_path}",
            ],
            strict=False,
        )
        if cfg.train is not None:
            train(cfg, work_dir=dir_path)
        if cfg.eval is not None:
            evaluate(cfg, work_dir=dir_path)
