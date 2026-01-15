class IrisError(Exception):
    """Base exception for the Iris library."""
    pass

class TaskError(IrisError):
    """Raised when task type cannot be determined."""
    pass

class DataError(IrisError):
    """Raised when input data invalid (e.g., empty, missing cols)."""
    pass

class DataLoadingError(Exception):
    pass