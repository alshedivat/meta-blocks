"""Training loop for meta-learning."""

import logging
import os
import random
from typing import Optional, Tuple

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
    meta_learner: MetaLearner,
    feed_batch: Tuple[common.FeedList],
    *,
    sess: Optional[tf.Session] = None,
    **sess_kwargs,
):
    """Performs one meta-training step.

    Parameters
    ----------
    meta_learner : MetaLearner

    feed_batch : tuple of FeedLists

    sess : tf.Session, optional

    Returns
    -------
    loss : float
        The loss value at the current training step.
    """
    if sess is None:
        sess = tf.get_default_session()

    # Make a training step and compute loss.
    losses, _ = sess.run(
        [meta_learner.meta_losses, meta_learner.meta_train_op],
        feed_dict=dict(sum(feed_batch, [])),
        **sess_kwargs,
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
        loss_ph = tf.placeholder(tf.float32, shape=())
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
        epoch = 0
        global_step = 0
        while global_step < cfg.train.max_steps:
            epoch += 1
            logger.info("-" * 48)
            logger.info(f"EPOCH: {epoch}")
            logger.info("-" * 48)

            feed_epoch = list(zip(*[td.epoch() for td in meta_learner.task_dists]))
            for batch, feed_batch in enumerate(feed_epoch):
                global_step += 1
                losses = train_step(meta_learner, feed_batch, sess=sess)

                # Log metrics.
                # TODO: create a utility function for logging.
                if global_step % cfg.train.log_interval == 0:
                    logger.info(f"global step: {global_step}")
                    logger.info(f"batch: {batch + 1}/{len(feed_epoch)}")
                    for loss, td in zip(losses, meta_learner.task_dists):
                        logger.info(f"{td.name}:")
                        logger.info(f"\tloss: {loss:.6f}")
                    for loss, writer in zip(losses, writers):
                        summary = sess.run(merged, feed_dict={loss_ph: loss})
                        writer.add_summary(summary, global_step)
                        writer.flush()

                # Save model.
                if global_step % cfg.train.save_interval == 0:
                    saver.save(checkpoint_number=global_step)

        # Save the final model.
        saver.save(checkpoint_number=global_step)
