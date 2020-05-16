"""Base classes for models."""


class Model:
    """The base class for models.

    Models consist of a collection of internally built :class:`Network` objects
    and define methods for building outputs (e.g., logits) and losses.
    The specific outputs and losses the model builds depend on the type
    of the model (i.e., classifier, regressor, policy, etc.).

    This base class is used for typing purposes.
    """
