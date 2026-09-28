"""Minimal binding shared by distinct inference schedules; no common algorithm."""


class InferenceEngine:
    """Bind a schedule to existing model state without creating another state store.

    Concrete engines intentionally expose different entry points. They do not
    share a generic run loop or own posterior values, components, or caches.
    """

    __slots__ = ("_model",)

    def __init__(self, model):
        self._model = model

    @property
    def model(self):
        """The live owner; engine creation does not initialize or mutate it."""
        return self._model
