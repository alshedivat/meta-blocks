"""Training loop for meta-learning."""

import logging
import os
import random
from typing import Optional

import numpy as np
import tensorflow.compat.v1 as tf
from omegaconf import DictConfig

from meta_blocks import common
from meta_blocks.adaptation.base import MetaLearner
from meta_blocks.experiment import utils

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


def train_step(
    meta_learner: MetaLearner, *, sess: Optional[tf.Session] = None, **kwargs
):
    """Performs one meta-training step.

    Parameters
    ----------
    meta_learner : MetaLearner

    sess : tf.Session, optional

    Returns
    -------
    loss : float
        The loss value at the current training step.
    """
    if sess is None:
        sess = tf.get_default_session()

    # Sample from the task distribution.
    feed_lists = [td.sample_task_feed() for td in meta_learner.task_dists]

    # Make a training step and compute loss.
    losses, _ = sess.run(
        [meta_learner.meta_losses, meta_learner.meta_train_op],
        feed_dict=dict(sum(feed_lists, [])),
        **kwargs,
    )

    return losses


def train(cfg: DictConfig, work_dir: Optional[str] = None, **session_kwargs):
    """Runs the training process for the provided config.

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

    # Setup the session.
    with utils.session(**session_kwargs) as sess:
        # Build and initialize.
        meta_learner = utils.build_and_initialize(cfg, mode=common.ModeKeys.TRAIN)

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

        # Setup checkpoint.
        checkpoint = tf.train.Checkpoint(
            model_state=meta_learner.model.trainable_parameters,
            optimizer=meta_learner.optimizer,
        )
        saver = tf.train.CheckpointManager(
            checkpoint, directory=work_dir, max_to_keep=5
        )

        # Do meta-learning iterations.
        logger.info("Training...")
        for i in range(cfg.train.max_steps):
            # Do multiple steps if the optimizer is multi-step.
            if cfg.train.optimizer.n is not None:
                losses = [
                    train_step(meta_learner, sess=sess)
                    for _ in range(cfg.train.optimizer.n)
                ]
                losses = list(map(np.mean, zip(*losses)))
            else:
                losses = train_step(meta_learner, sess=sess)

            # Log metrics.
            # TODO: create a utility function for logging.
            if i % cfg.train.log_interval == 0 or i + 1 == cfg.train.max_steps:
                logger.info(f"step: {i}")
                for loss, td in zip(losses, meta_learner.task_dists):
                    logger.info(f"{td.name}:")
                    if td.num_requested_labels:
                        logger.info(f"\trequested labels: {td.num_requested_labels}")
                    logger.info(f"\tloss: {loss:.6f}")
                for loss, td, writer in zip(losses, meta_learner.task_dists, writers):
                    feed_dict = {
                        loss_ph: loss,
                        label_budget_ph: td.num_requested_labels,
                    }
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
                for td, task in zip(meta_learner.task_dists, cfg.train.tasks):
                    td.expand(num_labeled_points=(task.labels_per_step * i), sess=sess)
                if cfg.train.do_reinit:
                    sess.run(tf.global_variables_initializer())
