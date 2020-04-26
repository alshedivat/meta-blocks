"""Entry point for running experiments."""

import logging
from multiprocessing import Lock, Process

import hydra
from omegaconf import DictConfig

from meta_blocks.experiment.eval import evaluate
from meta_blocks.experiment.train import train

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf/config.yaml", strict=False)
def main(cfg: DictConfig):
    cfg = cfg.meta_blocks
    processes = []
    lock = Lock()

    # Run evaluation process.
    if cfg.eval is not None:
        logger.debug("Starting evaluation...")
        eval_process = Process(target=evaluate, args=(cfg, lock), name="EVAL")
        eval_process.start()
        processes.append(eval_process)

    # Run training process.
    if cfg.train is not None:
        logger.debug("Starting training...")
        train_process = Process(target=train, args=(cfg, lock), name="TRAIN")
        train_process.start()
        processes.append(train_process)

    # Join processes.
    for p in processes:
        p.join(timeout=cfg.run.timeout)


if __name__ == "__main__":
    main()
