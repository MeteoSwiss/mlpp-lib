class MissingReparameterizationError(Exception):
    """Raised when a sampling function without 'rsample' is used in a context requiring reparameterization."""
    pass