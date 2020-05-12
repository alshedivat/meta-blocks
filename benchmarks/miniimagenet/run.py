"""The entry-point for running experiments."""

import logging

import hydra
from omegaconf import DictConfig

from meta_blocks.experiment.run import run_experiment

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf/config.yaml", strict=False)
def main(cfg: DictConfig):
    run_experiment(cfg)


if __name__ == "__main__":
    main()
