"""Evaluation loop for meta-learning."""

import collections
import logging
import os
import random
import time
from typing import Optional

import numpy as np
import tensorflow.compat.v1 as tf
from omegaconf import DictConfig

from meta_blocks import common
from meta_blocks.experiment import utils
from meta_blocks.experiment.utils import Experiment

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


def eval_step(
    exp: Experiment, repetitions: int = 1, sess: Optional[tf.Session] = None, **kwargs
):
    """Performs one evaluation step.

    Parameters
    ----------
    exp : Experiment
        The object that represents the experiment.
        Contains `meta_learners`, `samplers`, and `task_dists`.

    repetitions : int, optional (default: 1)
        Number of evaluation repetitions. Typically, set to 1.

    sess : tf.Session, optional
        The TF session used for executing the computation graph.

    Returns
    -------
    results : list of dicts
        List of dictionaries with eval metrics computed for each meta-learner.
    """
    if sess is None:
        sess = tf.get_default_session()

    # Re-initialize task distributions if samplers are stateful.
    for td in exp.task_dists:
        if td.sampler and td.sampler.stateful:
            td.initialize()

    # Sample from the task distribution.
    feed_lists = [
        td.sample_task_feed() for ml, td in zip(exp.meta_learners, exp.task_dists)
    ]

    # Do evaluation.
    results = []
    for ml, feed_list in zip(exp.meta_learners, feed_lists):
        results.append(collections.defaultdict(float))
        for _ in range(repetitions):
            # Perform predictions with adapted models on the query sets.
            preds_and_labels = sess.run(
                ml.preds_and_labels, feed_dict=dict(feed_list), **kwargs
            )
            # Evaluate predictions.
            avg_num_correct = 0.0
            for preds, labels in preds_and_labels:
                avg_num_correct += np.mean(preds == labels)
            results[-1]["acc"] += avg_num_correct / len(preds_and_labels)
        results[-1]["acc"] /= repetitions

    return results


def evaluate(cfg: DictConfig, work_dir: Optional[str] = None):
    """Runs the evaluation process for the provided config.

    Parameters
    ----------
    cfg : DictConfig
        The experiment configuration.

    work_dir : str, optional
        Working directory used for saving checkpoints, logs, etc.
        If None, it is set to `os.getcwd()`.
    """
    # Set working dir.
    if work_dir is None:
        work_dir = os.getcwd()

    # Set random seeds.
    random.seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    tf.set_random_seed(cfg.run.seed)

    with utils.session(gpu_allow_growth=True) as sess:
        # Build and initialize.
        exp = utils.build_and_initialize(cfg=cfg, mode=common.ModeKeys.EVAL)

        # Setup logging and saving.
        writers = [
            tf.summary.FileWriter(logdir=os.path.join(work_dir, task.log_dir))
            for task in cfg[common.ModeKeys.EVAL].tasks
        ]
        accuracy_ph = tf.placeholder(tf.float32, shape=())
        tf.summary.scalar("accuracy", accuracy_ph)
        merged = tf.summary.merge_all()

        # Run continuous eval.
        step = 0
        old_checkpoint = None
        while step + 1 < cfg.train.max_steps:
            # Get latest checkpoint.
            latest_checkpoint = tf.train.latest_checkpoint(work_dir)

            # If no change, wait and continue.
            if latest_checkpoint == old_checkpoint:
                time.sleep(cfg.eval.wait_time or 1)
                continue

            # Restore graph from the checkpoint.
            status = exp.checkpoint.restore(latest_checkpoint)
            status.assert_consumed().run_restore_ops()
            old_checkpoint = latest_checkpoint

            # Run evaluation.
            results = eval_step(exp, repetitions=cfg.eval.repetitions, sess=sess)

            # Log results.
            checkpoint_name = os.path.basename(latest_checkpoint)
            log = f"{'-' * 40}\n" f"evaluated: {checkpoint_name}"
            step = int(checkpoint_name.split("-")[1])
            for result, td in zip(results, exp.task_dists):
                log += f"\n{td.name} acc: {100 * result['acc']:.2f}"
            log += f"\n{'-' * 40}"
            logger.info(log)
            for result, td, writer in zip(results, exp.task_dists, writers):
                summary = sess.run(merged, feed_dict={accuracy_ph: result["acc"]})
                writer.add_summary(summary, step)
                writer.flush()
