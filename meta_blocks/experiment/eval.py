"""Evaluation loop for meta-learning."""

import collections
import logging
import os
import random
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
    exp: Experiment,
    *,
    repetitions: int = 1,
    sess: Optional[tf.Session] = None,
    **kwargs,
):
    """Performs one evaluation step.

    Parameters
    ----------
    exp : Experiment
        The object that represents the experiment.
        Contains `meta_learners`, `samplers`, and `task_dists`.

    repetitions : int (default: 1)
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

    results = [collections.defaultdict(list) for _ in exp.task_dists]

    # Do evaluation.
    for i, (ml, td) in enumerate(zip(exp.meta_learners, exp.task_dists)):
        for _ in range(repetitions):
            # Sample from the task distribution.
            feed_list = td.sample_task_feed()
            # Perform predictions with adapted models on the query sets.
            preds_and_labels_batch = sess.run(
                ml.preds_and_labels, feed_dict=dict(feed_list), **kwargs
            )
            # Evaluate predictions.
            # TODO: abstract this away into classification-specific metrics.
            accs = [100 * np.mean(p == l) for p, l in preds_and_labels_batch]
            results[i]["acc"].extend(accs)

    return results


def evaluate(cfg: DictConfig, work_dir: Optional[str] = None, **session_kwargs):
    """Runs the evaluation process for the provided config.

    Parameters
    ----------
    cfg : DictConfig
        The experiment configuration.

    work_dir : str, optional
        Working directory used for saving checkpoints, logs, etc.
        If None, it is set to `os.getcwd()`.

    **session_kwargs : kwargs
        Keyword arguments for configuring TF session
    """
    # Set working dir.
    if work_dir is None:
        work_dir = os.getcwd()

    # Set random seeds.
    random.seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    tf.set_random_seed(cfg.run.seed)

    with utils.session(**session_kwargs) as sess:
        # Build and initialize.
        exp = utils.build_and_initialize(cfg=cfg, mode=common.ModeKeys.EVAL)

        # Setup logging and saving.
        writers = [
            tf.summary.FileWriter(logdir=os.path.join(work_dir, task.log_dir))
            for task in cfg[common.ModeKeys.EVAL].tasks
        ]
        acc_phs = []
        for r in range(cfg.eval.repetitions):
            acc_ph = tf.placeholder(tf.float32, shape=())
            tf.summary.scalar(f"accuracy", acc_ph)
            acc_phs.append(acc_ph)
        merged = tf.summary.merge_all()

        # Run continuous eval.
        for latest_checkpoint in tf.train.checkpoints_iterator(work_dir):
            # Restore graph from the checkpoint.
            status = exp.checkpoint.restore(latest_checkpoint)
            status.assert_consumed().run_restore_ops()

            # Run evaluation.
            results = eval_step(exp, repetitions=cfg.eval.repetitions, sess=sess)

            # Log results.
            checkpoint_name = os.path.basename(latest_checkpoint)
            log = f"{'-' * 40}\n" f"evaluated: {checkpoint_name}"
            step = int(checkpoint_name.split("-")[1])
            for result, td in zip(results, exp.task_dists):
                acc_mean, acc_std = np.mean(result["acc"]), np.std(result["acc"])
                log += f"\n{td.name} acc: {acc_mean:.2f} ± {acc_std:.2f}"
            log += f"\n{'-' * 40}"
            logger.info(log)
            for result, td, writer in zip(results, exp.task_dists, writers):
                summary = sess.run(
                    merged,
                    feed_dict={
                        acc_ph: acc_val
                        for acc_ph, acc_val in zip(acc_phs, result["acc"])
                    },
                )
                writer.add_summary(summary, step)
                writer.flush()

            # Exit evaluation loop if done.
            if step + 1 >= cfg.train.max_steps:
                break
