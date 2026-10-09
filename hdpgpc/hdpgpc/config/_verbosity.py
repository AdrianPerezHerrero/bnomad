"""User-facing inference output levels and legacy boolean aliases."""


def normalize_verbosity(value):
    if value is True or (type(value).__name__ == 'bool_' and bool(value)):
        return 'debug'
    if value is False or (type(value).__name__ == 'bool_' and not bool(value)):
        return 'progress'
    if isinstance(value, str) and value in ('silent', 'progress', 'debug'):
        return value
    raise ValueError("verbose must be 'silent', 'progress', 'debug', or a boolean")
