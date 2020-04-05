"""Evaluation loop for meta-learning."""

import collections
import logging
import os
import random
import time

import numpy as np
import tensorflow.compat.v1 as tf

from meta_blocks import common, datasets
from meta_blocks.experiment import utils

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()

# Disable deprecation warnings.
tf.get_logger().setLevel(logging.ERROR)


def eval_step(cfg, exp, sess, **kwargs):
    """Performs one evaluation step.

    Parameters
    ----------
    cfg : Type and default value.
        The description string.

    exp : Type and default value.
        The description string.

    sess : Type and default value.
        The description string.

    kwargs

    Returns
    -------
    results : Type and default value.
        The description string.
    """
    # Re-initialize task distributions if samplers are stateful.
    for td in exp.task_dists:
        if td.sampler.stateful:
            td.initialize(sess)

    # Sample from the task distribution.
    feed_lists = [
        td.sample_task_feed() + ml.get_feed_list(**cfg.train.adapt)
        for ml, td in zip(exp.meta_learners, exp.task_dists)
    ]

    # Do evaluation.
    results = []
    for ml, feed_list in zip(exp.meta_learners, feed_lists):
        results.append(collections.defaultdict(float))
        for _ in range(cfg.eval.repetitions):
            # Perform predictions with adapted models on the query sets.
            preds_and_labels = sess.run(
                ml.preds_and_labels, feed_dict=dict(feed_list), **kwargs
            )
            # Evaluate predictions.
            avg_num_correct = 0.0
            for preds, labels in preds_and_labels:
                avg_num_correct += np.mean(preds == labels)
            results[-1]["acc"] += avg_num_correct / len(preds_and_labels)
        results[-1]["acc"] /= cfg.eval.repetitions

    return results


def evaluate(cfg, lock=None):
    """Runs the evaluation process for the provided config."""
    # Set random seeds.
    random.seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    tf.set_random_seed(cfg.run.seed)

    # Get categories.
    categories = datasets.get_categories(cfg.data.name, **cfg.data.read_config)

    with utils.session(gpu_allow_growth=True) as sess:
        # Build and initialize.
        if lock is not None:
            lock.acquire()
        exp = utils.build_and_initialize(
            cfg=cfg, sess=sess, categories=categories, mode=common.ModeKeys.EVAL
        )
        if lock is not None:
            lock.release()

        # Setup logging and saving.
        writers = [
            tf.summary.FileWriter(logdir=os.path.join(os.getcwd(), task.log_dir))
            for task in cfg[common.ModeKeys.EVAL].tasks
        ]
        accuracy_ph = tf.placeholder(tf.float32, shape=())
        tf.summary.scalar("accuracy", accuracy_ph)
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        # Run continuous eval.
        step = 0
        old_checkpoint = None
        while step + 1 < cfg.train.max_steps:
            # Get latest checkpoint.
            new_checkpoint = tf.train.latest_checkpoint(os.getcwd())

            # If no change, wait and continue.
            if new_checkpoint == old_checkpoint:
                time.sleep(cfg.eval.wait_time)
                continue

            # Restore graph from the checkpoint.
            saver.restore(sess, new_checkpoint)
            old_checkpoint = new_checkpoint

            # Run evaluation.
            results = eval_step(cfg, exp, sess)

            # Log results.
            step = int(os.path.basename(new_checkpoint).split("-")[1])
            log = f"EVAL - step: {step}"
            for result, td in zip(results, exp.task_dists):
                log += f" - {td.name} acc: {100 * result['acc']:.2f}"
            logger.info(log)
            for result, td, writer in zip(results, exp.task_dists, writers):
                summary = sess.run(merged, feed_dict={accuracy_ph: result["acc"]})
                writer.add_summary(summary, step)
                writer.flush()
