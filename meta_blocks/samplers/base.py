"""Base classes and functionality for sampling."""

import abc

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from typing import Tuple

from ..tasks import SupervisedTask

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class Sampler(abc.ABC):
    """Abstract class for sampling instances to be labeled."""

    stateful = False

    def __init__(self, stratified=False, name="Sampler"):
        """
        Instantiates a Sampler.

        Args:
            stratified: bool
            name: str
        """
        self.stratified = stratified
        self.name = name

        self._built = False

    def build(self, *args, **kwargs):
        """Builds sampler internals."""
        if not self._built:
            with tf.name_scope(self.name):
                self._build(*args, **kwargs)
            self._built = True
        return self

    @staticmethod
    def select_indices(size, scores, indices=None, soft=False) -> tf.Tensor:
        """Selects indices of the instances to label given the scores."""
        if soft:
            sampler = tfp.distributions.Uniform()
            z = -tf.math.log(-tf.math.log(sampler.sample(tf.shape(scores))))
            scores = tf.add(scores, z)
        with tf.control_dependencies([tf.assert_greater(tf.size(scores), 0)]):
            size = tf.minimum(size, tf.size(scores))
        _, selected_indices = tf.nn.top_k(scores, k=size)
        if indices is not None:
            selected_indices = tf.gather(indices, selected_indices, axis=0)
        return tf.cast(selected_indices, tf.int32)

    @staticmethod
    def stratify_by_cluster(size, clusters, parallel_iterations=8):
        unique_clusters, _, cluster_counts = tf.unique_with_counts(clusters)
        num_unique_clusters = tf.size(unique_clusters)

        def cond_fn(size_left, _cluster_counts_left, _cluster_sizes):
            return tf.greater(size_left, 0)

        def body_fn(size_left, cluster_counts_left, cluster_sizes):
            # Determine available clusters.
            cluster_mask = tf.greater(cluster_counts_left, 0)
            available_clusters = tf.where(cluster_mask)[:, 0]
            # Uniformly select clusters from available.
            indices = tf.random.uniform(
                dtype=tf.int32,
                shape=(size_left,),
                maxval=tf.size(available_clusters)
            )
            cluster_indices = tf.gather(available_clusters, indices, axis=0)
            cluster_sizes = tf.tensor_scatter_nd_add(
                cluster_sizes,
                indices=tf.expand_dims(cluster_indices, -1),
                updates=tf.ones_like(cluster_indices, dtype=tf.int32)
            )
            # Truncate cluster sizes as necessary.
            cluster_sizes = tf.minimum(cluster_sizes, cluster_counts)
            cluster_counts_left = cluster_counts - cluster_sizes
            size_left = size - tf.reduce_sum(cluster_sizes)
            return size_left, cluster_counts_left, cluster_sizes

        # Ideal stratification.
        min_size = tf.math.floordiv(size, num_unique_clusters)
        cluster_sizes_init = tf.tile([min_size], [num_unique_clusters])
        cluster_sizes_init = tf.minimum(cluster_sizes_init, cluster_counts)
        cluster_counts_left_init = cluster_counts - cluster_sizes_init
        size_left_init = size - tf.reduce_sum(cluster_sizes_init)

        # Keep sampling uniformly from available clusters to add up to size.
        _, _, cluster_sizes = tf.while_loop(
            cond=cond_fn,
            body=body_fn,
            loop_vars=[
                size_left_init, cluster_counts_left_init, cluster_sizes_init
            ],
            back_prop=False,
            parallel_iterations=parallel_iterations,
            name="stratified-sampling",
        )

        return cluster_sizes, unique_clusters

    @staticmethod
    def select_indices_stratified(
        size, scores, clusters, indices=None, soft=False, parallel_iterations=8,
    ) -> tf.Tensor:
        """Selects indices of the instances to label given the scores."""
        # size_per_cluster: <int32> [num_unique_clusters].
        # unique_clusters: <int32> [num_unique_clusters].
        size_per_cluster, unique_clusters = Sampler.stratify_by_cluster(
            size, clusters, parallel_iterations=parallel_iterations
        )

        def cond_fn(step, _unused_indices):
            return tf.less(step, tf.size(size_per_cluster))

        def body_fn(step, selected_indices):
            cluster_mask = tf.equal(clusters, unique_clusters[step])
            cluster_indices = tf.where(cluster_mask)[:, 0]
            cluster_scores = tf.gather(scores, cluster_indices, axis=0)
            selected_idx = tf.cond(
                pred=tf.greater(size_per_cluster[step], 0),
                true_fn=lambda: Sampler.select_indices(
                    size=size_per_cluster[step],
                    scores=cluster_scores,
                    indices=cluster_indices,
                    soft=soft,
                ),
                false_fn=lambda: tf.constant([], dtype=tf.int32)
            )
            return [tf.add(step, 1), selected_indices.write(step, selected_idx)]

        # Select indices for each cluster cluster.
        _, selected_indices_ta = tf.while_loop(
            cond=cond_fn,
            body=body_fn,
            loop_vars=[
                tf.constant(0),
                tf.TensorArray(
                    dtype=tf.int32,
                    infer_shape=False,
                    size=tf.size(unique_clusters),
                ),
            ],
            back_prop=False,
            parallel_iterations=parallel_iterations,
            name="stratified-index-selection",
        )

        selected_indices = selected_indices_ta.concat()
        selected_indices = tf.reshape(selected_indices, shape=(size,))
        if indices is not None:
            selected_indices = tf.gather(indices, selected_indices, axis=0)

        return selected_indices

    @abc.abstractmethod
    def _build(self, *args, **kwargs):
        """Builds sampler internals."""
        raise NotImplementedError("Abstract Method")

    @abc.abstractmethod
    def compute_scores(self, task: SupervisedTask, **kwargs) -> tf.Tensor:
        """Computes scores of each unlabeled instance in the support set."""
        raise NotImplementedError("Abstract Method")

    @abc.abstractmethod
    def select_labeled(
        self,
        size: int,
        sess: tf.Session,
        tasks: Tuple[SupervisedTask],
        **kwargs
    ) -> Tuple[np.ndarray]:
        """Return an actively selected labeled data points from the dataset."""
        raise NotImplementedError("Abstract Method")
