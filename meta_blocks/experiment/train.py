"""Training loop for meta-learning."""

import logging
import os
import random

import numpy as np
import tensorflow.compat.v1 as tf

from meta_blocks import common, datasets
from meta_blocks.experiment import utils

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


def train_step(cfg, exp, sess, **kwargs):
    """Performs one meta-training step.

    Parameters
    ----------
    cfg : OmegaConf
        The experiment configuration.

    exp : Experiment
        The object that represents the experiment.
        Contains `meta_learners`, `samplers`, and `task_dists`.

    sess : tf.Session
        The TF session used for executing the computation graph.

    Returns
    -------
    losses : list of floats
        Loss functions computed for each meta-learner.
    """
    # Sample from the task distribution.
    feed_lists = [
        td.sample_task_feed() + ml.get_feed_list(**cfg.train.adapt)
        for ml, td in zip(exp.meta_learners, exp.task_dists)
    ]

    # Train and compute losses.
    losses = []
    for ml, feed_list in zip(exp.meta_learners, feed_lists):
        loss, _ = sess.run([ml.loss, ml.train_op], feed_dict=dict(feed_list), **kwargs)
        losses.append(loss)

    return losses


def train(cfg, lock=None, work_dir=None):
    """Runs the training process for the provided config.

    Parameters
    ----------
    cfg : OmegaConf
        The experiment configuration.

    lock : Lock
        An object used for synchronizing training and evaluation processes.

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

    # Get categories.
    logger.debug(f"Reading {cfg.data.name} categories...")
    categories = datasets.get_categories(cfg.data.name, **cfg.data.read_config)

    # Setup the session.
    with utils.session(gpu_allow_growth=True) as sess:
        # Build and initialize.
        exp = utils.build_and_initialize(
            cfg=cfg, sess=sess, categories=categories, mode=common.ModeKeys.TRAIN
        )

        # Setup logging and saving.
        writers = [
            tf.summary.FileWriter(logdir=os.path.join(work_dir, task.log_dir))
            for task in cfg[common.ModeKeys.TRAIN].tasks
        ]
        label_budget_ph = tf.placeholder(tf.int32, shape=())
        loss_ph = tf.placeholder(tf.float32, shape=())
        tf.summary.scalar("label_budget", label_budget_ph)
        tf.summary.scalar("loss", loss_ph)
        merged = tf.summary.merge_all()
        saver = tf.train.CheckpointManager(
            exp.checkpoint, directory=work_dir, max_to_keep=1
        )

        # Do meta-learning iterations.
        if lock is not None:
            lock.acquire()
        logger.info("Training...")
        for i in range(cfg.train.max_steps):
            # Training step.
            losses = train_step(cfg, exp, sess)
            # Log metrics.
            if i % cfg.train.log_interval == 0 or i + 1 == cfg.train.max_steps:
                log = f"step: {i}"
                for loss, td in zip(losses, exp.task_dists):
                    log += f"\n\trequested labels: {td.requested_labels}"
                    log += f"\n\t{td.name} loss: {loss:.6f}"
                logger.info(log)
                for loss, td, writer in zip(losses, exp.task_dists, writers):
                    feed_dict = {loss_ph: loss, label_budget_ph: td.requested_labels}
                    summary = sess.run(merged, feed_dict=feed_dict)
                    writer.add_summary(summary, i)
                    writer.flush()
            # Save model.
            if i % cfg.train.save_interval == 0 or i + 1 == cfg.train.max_steps:
                saver.save(checkpoint_number=i)
            # Update task distribution (if necessary).
            # TODO: make this more flexible.
            if (
                cfg.train.budget_interval is not None
                and i % cfg.train.budget_interval == 0
            ):
                for td, task in zip(exp.task_dists, cfg.train.tasks):
                    td.expand(num_labeled_points=(task.labels_per_step * i), sess=sess)
                if cfg.train.do_reinit:
                    sess.run(tf.global_variables_initializer())
        if lock is not None:
            lock.release()
