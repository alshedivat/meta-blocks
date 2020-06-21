"""Evaluation loop for meta-learning."""

import logging
import os
import random
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import tensorflow.compat.v1 as tf
from omegaconf import DictConfig

from meta_blocks import common
from meta_blocks.adaptation.base import MetaLearner
from meta_blocks.experiment import utils
from meta_blocks.experiment.metrics import (
    build_metrics_and_summaries,
    get_layout_summary,
)

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


def eval_step(
    meta_learner: MetaLearner,
    *,
    tasks: Tuple[DictConfig, ...],
    metric_fns: Dict[str, Callable],
    repetitions: int = 1,
    sess: Optional[tf.Session] = None,
    **kwargs,
):
    """Performs one evaluation step.

    Parameters
    ----------
    meta_learner : MetaLearner

    tasks : tuple of DictConfigs
        Configurations for the evaluation tasks.

    metric_fns : dict of functions
        A dictionary of metric functions that map a list of (prediction, label)
        tuples to a numeric value.

    repetitions : int (default: 1)
        Number of evaluation repetitions. Typically, set to 1.

    sess : tf.Session, optional
        The TF session used for executing the computation graph.

    Returns
    -------
    metric_values : dict of dicts
    """
    if sess is None:
        sess = tf.get_default_session()

    # Re-initialize task distributions if samplers are stateful.
    for td in meta_learner.task_dists:
        if td.sampler and td.sampler.stateful:
            td.initialize()

    # Do evaluation.
    metric_values = defaultdict(dict)
    for preds_and_labels_batch_tf, td, t in zip(
        meta_learner.preds_and_labels, meta_learner.task_dists, tasks
    ):
        # Compute preds_and_labels and labels.
        preds_and_labels_np = []
        for _ in range(repetitions):
            # Sample from the task distribution.
            feed_list = td.sample_task_feed()
            # Predict query set labels using adapted model.
            # TODO: compute all pre/post metrics on support/query.
            preds_and_labels_batch_np = sess.run(
                [pl["post"]["query"] for pl in preds_and_labels_batch_tf],
                feed_dict=dict(feed_list),
                **kwargs,
            )
            preds_and_labels_np.extend(preds_and_labels_batch_np)
        # Compute compute metrics.
        task_scope = f"{t.set_name}/{t.regime}"
        for metric_name, metric_fn in metric_fns.items():
            metric_values[task_scope][metric_name] = metric_fn(preds_and_labels_np)

    return metric_values


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
        meta_learner = utils.build_and_initialize(cfg, mode=common.ModeKeys.EVAL)

        # Setup checkpoint.
        checkpoint = tf.train.Checkpoint(
            model_state=meta_learner.model.trainable_parameters,
            optimizer=meta_learner.optimizer,
        )

        # Build metrics.
        metric_fns, metric_phs, summary_ops = build_metrics_and_summaries(
            metrics=cfg.eval.metrics, tasks=cfg.eval.tasks
        )

        # Setup TensorBoard layout.
        layout_summary = get_layout_summary(
            metrics=cfg.eval.metrics, tasks=cfg.eval.tasks
        )
        writers = {}
        for log_dir in sorted(set(t.log_dir for t in cfg.eval.tasks)):
            writers[log_dir] = tf.summary.FileWriter(
                logdir=os.path.join(work_dir, log_dir)
            )
            writers[log_dir].add_summary(layout_summary)

        # Run continuous eval.
        for latest_checkpoint in tf.train.checkpoints_iterator(work_dir):
            # Restore graph from the checkpoint.
            status = checkpoint.restore(latest_checkpoint)
            status.assert_consumed().run_restore_ops()

            # Run evaluation.
            metric_values = eval_step(
                meta_learner,
                tasks=cfg.eval.tasks,
                metric_fns=metric_fns,
                repetitions=cfg.eval.repetitions,
                sess=sess,
            )

            # Log results and build feed list for saving summaries.
            # TODO: create a utility function for logging.
            summary_feed_lists = defaultdict(list)
            checkpoint_name = os.path.basename(latest_checkpoint)
            logger.info(f"{'-' * 50}")
            logger.info(f"evaluated: {checkpoint_name}")
            step = int(checkpoint_name.split("-")[1])
            for m in cfg.eval.metrics:
                for t in cfg.eval.tasks:
                    task_scope = f"{t.set_name}/{t.regime}"
                    # Add metric to the log.
                    mean_value, ci_value = metric_values[task_scope][m.name]
                    ci_delta = (ci_value[1] - ci_value[0]) / 2.0
                    logger.info(
                        f"{task_scope}/{m.name} (CI {m.ci:.0f}%):".ljust(35)
                        + f"{mean_value: >5.2f} ± {ci_delta: >5.2f}"
                    )
                    # Add items to the summary feed list.
                    mean_ph, lower_ph, upper_ph = metric_phs[task_scope][m.name]
                    summary_feed_lists[t.log_dir].extend(
                        [
                            (mean_ph, mean_value),
                            (lower_ph, ci_value[0]),
                            (upper_ph, ci_value[1]),
                        ]
                    )
            logger.info(f"{'-' * 50}")

            # Log summaries.
            for log_dir, writer in writers.items():
                feed_dict = dict(summary_feed_lists[log_dir])
                summaries = sess.run(summary_ops[log_dir], feed_dict=feed_dict)
                for summary in summaries:
                    writer.add_summary(summary, step)
                writer.flush()

            # Exit evaluation loop if done.
            if step + 1 >= cfg.train.max_steps:
                break

        # Close summary writers.
        for writer in writers.values():
            writer.close()
