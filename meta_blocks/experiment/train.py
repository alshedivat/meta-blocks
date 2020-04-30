"""Training loop for meta-learning."""

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


def train_step(exp: Experiment, sess: Optional[tf.Session] = None, **kwargs):
    """Performs one meta-training step.

    Parameters
    ----------
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
    if sess is None:
        sess = tf.get_default_session()

    # Sample from the task distribution.
    feed_lists = [
        td.sample_task_feed() for ml, td in zip(exp.meta_learners, exp.task_dists)
    ]

    # Train and compute losses.
    losses = []
    for ml, feed_list in zip(exp.meta_learners, feed_lists):
        loss, _ = sess.run([ml.loss, ml.train_op], feed_dict=dict(feed_list), **kwargs)
        losses.append(loss)

    return losses


def train(cfg: DictConfig, work_dir: Optional[str] = None):
    """Runs the training process for the provided config.

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

    # Setup the session.
    with utils.session(gpu_allow_growth=True) as sess:
        # Build and initialize.
        exp = utils.build_and_initialize(cfg=cfg, mode=common.ModeKeys.TRAIN)

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
            exp.checkpoint, directory=work_dir, max_to_keep=5
        )

        # Do meta-learning iterations.
        logger.info("Training...")
        for i in range(cfg.train.max_steps):
            # Training step.
            # Do multiple steps if the optimizer is multi-step.
            if "multistep" in cfg.train.optimizer.name:
                losses = [
                    train_step(exp, sess=sess) for _ in range(cfg.train.optimizer.n)
                ]
                losses = list(map(np.mean, zip(*losses)))
            else:
                losses = train_step(exp, sess=sess)
            # Log metrics.
            if i % cfg.train.log_interval == 0 or i + 1 == cfg.train.max_steps:
                log = f"step: {i}"
                for loss, td in zip(losses, exp.task_dists):
                    if td.num_requested_labels:
                        log += f"\nrequested labels: {td.num_requested_labels}"
                    log += f"\n{td.name} loss: {loss:.6f}"
                logger.info(log)
                for loss, td, writer in zip(losses, exp.task_dists, writers):
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
                for td, task in zip(exp.task_dists, cfg.train.tasks):
                    td.expand(num_labeled_points=(task.labels_per_step * i), sess=sess)
                if cfg.train.do_reinit:
                    sess.run(tf.global_variables_initializer())
