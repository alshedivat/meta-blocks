"""Base models for meta-learning."""

import abc
import logging

import tensorflow.compat.v1 as tf

__all__ = ["Model"]

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class Model(abc.ABC):
    """Abstract base class for models.

    Each model is supposed to implement _build_embeddings and _build_logits
    that build symbolic tensors on top provided of input placeholders.

    Parameters of the sub-graph that computes embeddings are assumed to be
    "global" (i.e., shared across tasks), and parameters of the sub-graph that
    computes logits are assumed to be "adaptable" (i.e., per-task adjustable).

    Parameters
    ----------
    num_classes : int
        The number of classes.

    name : str, optional (default="Model")
        The description string.

    global_embeddings : bool, optional (default=False)
        The description string.
    """

    def __init__(
        self, num_classes, name="Model", global_embeddings=False, **kwargs
    ):
        """


        """
        self.global_embeddings = global_embeddings
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.name = name

        # Placeholders and ops.
        self.inputs_ph = None
        self.labels_ph = None
        self.embeddings = None
        self.logits = None
        self.loss = None
        self.preds = None

        # Internal.
        self._custom_getter = None

        self.built = False

    @property
    def custom_getter(self):
        return self._custom_getter

    @custom_getter.setter
    def custom_getter(self, custom_getter):
        self._custom_getter = custom_getter

    @staticmethod
    @abc.abstractmethod
    def get_data_placeholders():
        """Returns data placeholders expected by the model.
        Must be implemented in a subclass.
        """
        raise NotImplementedError("Abstract Method")

    @property
    def adaptable_parameters(self):
        """Returns a dict of adaptable parameters."""
        variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=f"{self.name}/adaptable"
        )
        return {var.name.split(":")[0]: var for var in variables}

    @property
    def global_parameters(self):
        """Returns a dict of global parameters."""
        variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=f"{self.name}/global"
        )
        return {var.name.split(":")[0]: var for var in variables}

    @property
    def parameters(self):
        """Returns a dict of all trainable parameters."""
        variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name
        )
        return {var.name.split(":")[0]: var for var in variables}

    @abc.abstractmethod
    def build(self):
        """Builds the model graph for prediction.
        Must be implemented in a subclass.
        """
        raise NotImplementedError("Abstract Method")

    def build_embeddings(self, inputs_ph, reuse=True):
        """Builds intermediate representations used as input embeddings."""
        if self.global_embeddings:
            scope = f"{self.name}/global"
        else:
            scope = f"{self.name}/adaptable"
        with tf.variable_scope(
            scope, custom_getter=self.custom_getter, reuse=reuse
        ):
            # <float32> [num_inputs, emb_size].
            embeddings = self._build_embeddings(inputs_ph)
        return embeddings

    @abc.abstractmethod
    def _build_embeddings(self, input_ph):
        """Builds a part of the model graph for computing input embeddings.
        Must be implemented in a subclass.
        """
        raise NotImplementedError("Abstract Method")

    def build_logits(self, embeddings, reuse=True):
        """Builds logits on top the provided embeddings."""
        scope = f"{self.name}/adaptable"
        with tf.variable_scope(
            scope, custom_getter=self.custom_getter, reuse=reuse
        ):
            # <float32> [num_inputs, num_classes].
            logits = self._build_logits(embeddings)
        return logits

    @abc.abstractmethod
    def _build_logits(self, embeddings):
        """Builds a part of the model graph for computing output logits.
        Must be implemented in a subclass.
        """
        raise NotImplementedError("Abstract Method")

    def build_loss(
        self,
        inputs_ph,
        labels_ph,
        num_classes=None,
        use_sparse_softmax=False,
        reuse=True,
    ):
        """Builds the model loss on top provided data placeholders."""
        with tf.name_scope(self.name):
            # Build embeddings.
            # <float32> [num_inputs, emb_size].
            embeddings = self.build_embeddings(inputs_ph, reuse=reuse)

            # Build logits.
            # <float32> [num_inputs, num_classes].
            logits = self.build_logits(embeddings, reuse=reuse)

            # Build loss.
            if use_sparse_softmax:
                # <float32> [num_inputs].
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels_ph, logits=logits
                )
            else:
                num_classes = num_classes or self.num_classes
                # Note: using dense softmax by default b/c sparse doesn't allow
                #       2nd order gradients for now (as of TF 1.14.0).
                # <float32> [num_inputs, num_classes].
                labels_onehot = tf.one_hot(labels_ph, num_classes)
                # <float32> [num_inputs].
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=labels_onehot, logits=logits
                )

        return loss

    def build_predictions(self, inputs_ph, reuse=True):
        """Builds the model loss on top provided data placeholders."""
        with tf.name_scope(self.name):
            # Build embeddings.
            # <float32> [num_inputs, emb_size].
            embeddings = self.build_embeddings(inputs_ph, reuse=reuse)

            # Build logits.
            # <float32> [num_inputs, num_classes].
            logits = self.build_logits(embeddings, reuse=reuse)

            # Build predictions.
            # <float32> [num_inputs].
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)

        return preds

    def build_margin_scores(self, inputs_ph, reuse=True):
        """Builds the model logits on top provided data placeholders."""
        with tf.name_scope(self.name):
            # Build embeddings.
            # <float32> [num_inputs, emb_size].
            embeddings = self.build_embeddings(inputs_ph, reuse=reuse)

            # Build logits.
            # <float32> [num_inputs, num_classes].
            logits = self.build_logits(embeddings, reuse=reuse)

            # Build probs and scores.
            # <float32> [num_inputs, num_classes].
            probs = tf.nn.softmax(logits, axis=-1)
            # <float32> [num_inputs].
            scores = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=probs
            )

        return scores


class FeedForwardModel(Model):
    """The base class for feed-forward models."""

    def __init__(self, num_classes, name="FeedForwardModel", **kwargs):
        Model.__init__(self, num_classes, name=name, **kwargs)

    def build(self):
        """Builds all parametric components of the model graph."""
        # Get default data placeholders.
        with tf.name_scope(self.name):
            self.inputs_ph, self.labels_ph = self.get_data_placeholders()

        # Build embeddings.
        # <float32> [num_inputs, emb_size].
        self.embeddings = self.build_embeddings(self.inputs_ph, reuse=False)

        # Build logits.
        # <float32> [num_inputs, num_classes].
        self.logits = self.build_logits(self.embeddings, reuse=False)

        self.built = True
        return self


class ProtoModel(Model):
    """The base class for models that use prototypes to compute logits."""

    def __init__(self, name="ProtoModel", **kwargs):
        Model.__init__(self, name=name, **kwargs)

        # Internals.
        self._prototypes = None

    @property
    def prototypes(self):
        return self._prototypes

    @prototypes.setter
    def prototypes(self, prototypes):
        self._prototypes = prototypes

    def build(self):
        """Builds all parametric components of the model graph."""
        # Get default data placeholders.
        with tf.name_scope(self.name):
            self.inputs_ph, self.labels_ph = self.get_data_placeholders()

        # Build embeddings.
        # <float32> [num_inputs, emb_size].
        self.embeddings = self.build_embeddings(self.inputs_ph, reuse=False)

        self.built = True
        return self

    def _build_logits(self, embeddings):
        """Builds logits using distances between inputs and prototypes."""
        # TODO: generalize to other types of distances.
        # <float32> [num_inputs, num_classes].
        square_dists_emb_proto = tf.reduce_sum(
            tf.square(
                # <float32> [num_inputs, 1, emb_size].
                tf.expand_dims(embeddings, 1)
                -
                # <float32> [1, num_classes, emb_size].
                tf.expand_dims(self.prototypes, 0)
            ),
            axis=-1,
        )
        # <float32> [num_inputs, num_classes].
        return tf.nn.log_softmax(-square_dists_emb_proto, axis=-1)
