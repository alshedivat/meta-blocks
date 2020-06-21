"""Utility functions for running experiments."""

import contextlib
import datetime
import logging
import re
from typing import Optional

import colorlog
import tensorflow.compat.v1 as tf

from meta_blocks import (
    adaptation,
    common,
    datasets,
    models,
    networks,
    optimizers,
    samplers,
    tasks,
)

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


@contextlib.contextmanager
def session(
    gpu_ids: Optional[str] = None,
    gpu_allow_growth: bool = True,
    log_device_placement: bool = False,
):
    """Sets up an experiment and returns a tf.Session.

    Parameters
    ----------
    gpu_ids : str, optional
        GPUs that will be made visible.

    gpu_allow_growth : bool (default: True)
        Used to configure tf.Session.
        See tf.ConfigProto.gpu_options.allow_growth.

    log_device_placement : bool (default: False)
        Used for debugging tensor placement on devices.
        See tf.ConfigProto.log_device_placement.
    """
    # Reset TF graph.
    tf.reset_default_graph()

    # Create and configure a tf.Session.
    config = tf.ConfigProto()
    if gpu_ids is not None:
        config.gpu_options.visible_device_list = gpu_ids
    if gpu_allow_growth:
        config.gpu_options.allow_growth = True
    if log_device_placement:
        config.log_device_placement = True
    sess = tf.Session(config=config)

    try:
        with sess.as_default():
            yield sess
    finally:
        sess.close()


def build_and_initialize(cfg, mode=common.ModeKeys.TRAIN):
    """Builds and initializes all parts of the graph.

    Parameters
    ----------
    cfg : OmegaConf
        The experiment configuration.

    mode : str, optional (default: common.ModeKeys.TRAIN)
        Defines the mode of the computation graph (TRAIN or EVAL).
        Note: this is likely to be removed from the API down the line.

    Returns
    -------
    exp : Experiment
        An object that represents the experiment.
        Contains `meta_learners`, `samplers`, and `task_dists`.
    """
    sess = tf.get_default_session()

    # Build the data source.
    data_source = datasets.get_data_source(
        name=cfg.data.name, **cfg.data.source
    ).build()

    # Build meta-datasets.
    meta_datasets = {}
    for set_name in set(task.set_name for task in cfg[mode].tasks):
        meta_datasets[set_name] = datasets.get_meta_dataset(
            name=cfg.data.name,
            data_sources=data_source[set_name],
            **cfg[mode].meta_dataset,
        ).build()

    # Build task distributions.
    task_dists = []
    for task in cfg[mode].tasks:
        task_dist = tasks.get_distribution(
            meta_dataset=meta_datasets[task.set_name],
            name_suffix=f"{task.set_name}_{task.regime}",
            sampler_config=task.sampler,
            **task.config,
        ).build()
        task_dists.append(task_dist)

    # Build model.
    network_builder = networks.get(**cfg.network)
    model_builder = models.get(
        input_shapes=data_source.data_shapes,
        input_types=data_source.data_types,
        num_classes=cfg[mode].meta_dataset.num_classes,
        network_builder=network_builder,
        **cfg[mode].model,
    )

    # Build optimizer.
    optimizer = optimizers.get(**cfg.train.optimizer)

    # Build meta-learner.
    meta_learner = adaptation.get(
        model_builder=model_builder,
        optimizer=optimizer,
        task_dists=task_dists,
        mode=mode,
        **cfg[mode].adapt,
    )

    # Variable initialization.
    if mode == common.ModeKeys.TRAIN:
        # Initialize all the variables in the graph.
        sess.run(tf.global_variables_initializer())
    else:  # mode == common.ModeKeys.EVAL:
        # Initialize only non-trainable variables.
        # Note: Trainable variables must be loaded from a checkpoint.
        #       Being explicit about which variables are initialized is better
        #       prevents weird side effects when we are unaware of some created
        #       variables that are silently initialized at evaluation time.
        sess.run(tf.variables_initializer(meta_learner.non_trainable_parameters))

    # Initialize task distributions.
    for task, task_dist in zip(cfg[mode].tasks, task_dists):
        sampler = None
        if task.sampler is not None:
            sampler = samplers.get(**task.sampler)
            sampler.build(task_dist=task_dist, meta_learner=meta_learner)
        task_dist.initialize(sampler=sampler)

    return meta_learner


class ExperimentFormatter(colorlog.ColoredFormatter):
    """Custom log formatter that nicely indents multiline experiment logs and
    format times relative to the start of the star time of the experiment.
    """

    def formatTime(self, record, datefmt=None):
        """
        Return the creation time of the specified LogRecord relative to the
        start time of the experiment as formatted text.
        """
        duration = datetime.datetime.utcfromtimestamp(record.relativeCreated / 1000)
        record.delta = duration.strftime("%H:%M:%S")
        return super(ExperimentFormatter, self).formatTime(record, datefmt)
