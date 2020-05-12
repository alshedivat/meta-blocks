"""Metric and utility functions used for evaluation."""
import logging
from collections import defaultdict
from typing import Callable, Tuple

import numpy as np
import scikits.bootstrap as boot
import tensorflow.compat.v1 as tf
from omegaconf import DictConfig
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.summary import v1 as summary_lib

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


def get_layout_summary(
    *,
    metrics: Tuple[DictConfig, ...],
    tasks: Tuple[DictConfig, ...],
    closed: bool = False,
):
    """Builds a summary that describes custom scalars layout for TensorBoard.

    At each evaluation step, we compute metrics for multiple tasks sampled from
    the same distribution and estimate the mean and std value of the metric.
    The layout specified by this function nicely groups metrics and defines
    margin charts that use fill area to visualize lower and upper bounds.

    Parameters
    ----------
    metrics : tuple of DictConfigs

    tasks : tuple of DictConfigs

    closed : bool (default: False)

    Returns
    -------
      A summary proto containing the layout.
    """
    set_names = sorted(set(t.set_name for t in tasks))
    task_regimes = sorted(set(t.regime for t in tasks))
    layout_summary = summary_lib.custom_scalar_pb(
        layout_pb2.Layout(
            category=[
                # Category for each metric.
                layout_pb2.Category(
                    title=m.name,
                    chart=[
                        # A chart for each type of the eval distribution.
                        layout_pb2.Chart(
                            title=f"{t_regime}/{m.name} (CI {m.ci:.0f}%)",
                            margin=layout_pb2.MarginChartContent(
                                series=[
                                    layout_pb2.MarginChartContent.Series(
                                        value=f"{s_name}/{t_regime}/{m.name}_mean/scalar_summary",
                                        lower=f"{s_name}/{t_regime}/{m.name}_lower/scalar_summary",
                                        upper=f"{s_name}/{t_regime}/{m.name}_upper/scalar_summary",
                                    )
                                    for s_name in set_names
                                ]
                            ),
                        )
                        for t_regime in task_regimes
                    ],
                    closed=closed,
                )
                for m in metrics
            ]
        )
    )
    return layout_summary


def build_metrics_and_summaries(
    metrics: Tuple[DictConfig, ...], tasks: Tuple[DictConfig, ...]
):
    """Builds metric placeholders, summaries, and functions for computing metrics."""
    metric_fns = dict()
    metric_phs = defaultdict(dict)
    summaries = defaultdict(list)
    for m in metrics:
        # Build metric functions.
        metric_fns[f"{m.name}"] = get_metric_stats_fn(
            metric_fn=get_metric_fn(m.name), ci=m.ci
        )
        for t in tasks:
            task_scope = f"{t.set_name}/{t.regime}"
            # Build metric placeholders and summaries.
            with tf.name_scope(f"{task_scope}"):
                metric_phs[f"{task_scope}"][f"{m.name}"] = []
                for stat in ["mean", "lower", "upper"]:
                    metric_ph = tf.placeholder(tf.float32, shape=())
                    metric_phs[f"{task_scope}"][f"{m.name}"].append(metric_ph)
                    summary = summary_lib.scalar(f"{m.name}_{stat}", metric_ph)
                    summaries[t.log_dir].append(summary)
    return metric_fns, metric_phs, summaries


def get_metric_stats_fn(metric_fn: Callable, ci: float = 68.0):
    """Returns a closure that computes metric mean and ci bounds."""

    def _metrics_stats_fn(preds_and_labels):
        # Slice predictions and labels into batches and compute metrics on them.
        metric_values = np.asarray(metric_fn(preds_and_labels))
        # Compute metric mean and CI using bootstrap.
        metric_mean = np.mean(metric_values)
        metric_ci = boot.ci(metric_values, np.mean, alpha=(1.0 - ci / 100.0))
        return metric_mean, metric_ci

    return _metrics_stats_fn


def accuracy(preds_and_labels_batch):
    """Computes accuracy from predictions and labels."""
    return [100.0 * np.mean(ps == ls) for ps, ls in preds_and_labels_batch]


def get_metric_fn(name):
    if name == "accuracy":
        return accuracy
    else:
        raise ValueError(f"Unknown metric: {name}")
