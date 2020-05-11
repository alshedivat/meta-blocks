"""The entry-point for running experiments."""

import logging
from multiprocessing import Process

from omegaconf import DictConfig

from meta_blocks.experiment.eval import evaluate
from meta_blocks.experiment.train import train

logger = logging.getLogger(__name__)


def run_experiment(cfg: DictConfig):
    cfg = cfg.meta_blocks
    processes = []

    # Run evaluation process.
    if cfg.eval is not None:
        logger.debug("Starting evaluation...")
        eval_process = Process(
            target=evaluate,
            kwargs={
                "cfg": cfg,
                "gpu_ids": str(cfg.compute.gpus.eval.ids),
                "gpu_allow_growth": cfg.compute.gpus.eval.allow_growth,
            },
            name="EVAL",
        )
        eval_process.start()
        processes.append(eval_process)

    # Run training process.
    if cfg.train is not None:
        logger.debug("Starting training...")
        train_process = Process(
            target=train,
            kwargs={
                "cfg": cfg,
                "gpu_ids": str(cfg.compute.gpus.train.ids),
                "gpu_allow_growth": cfg.compute.gpus.train.allow_growth,
            },
            name="TRAIN",
        )
        train_process.start()
        processes.append(train_process)

    # Join processes.
    for p in processes:
        p.join(timeout=cfg.run.timeout)
