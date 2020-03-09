"""Entry point for running experiments."""

import hydra
import logging

from multiprocessing import Process, Lock

from .experiment.train import train
from .experiment.eval import evaluate

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/config.yaml", strict=False)
def main(cfg):
    processes = []
    print(cfg.pretty())
    lock = Lock()

    # Run training process, if necessary.
    if cfg.eval is not None:
        logger.debug("Starting evaluation...")
        eval_process = Process(target=evaluate, args=(cfg, lock))
        eval_process.start()
        processes.append(eval_process)

    # Run training process, if necessary.
    if cfg.train is not None:
        logger.debug("Starting training...")
        train_process = Process(target=train, args=(cfg, lock))
        train_process.start()
        processes.append(train_process)

    # Join processes.
    for p in processes:
        p.join(timeout=cfg.run.timeout)


if __name__ == "__main__":
    main()
