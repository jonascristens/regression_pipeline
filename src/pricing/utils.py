from functools import wraps


def expand_args(func):
    @wraps(func)
    def wrapper(*args):
        kwargs = next((arg for arg in args if isinstance(arg, dict)), {})
        args = tuple(arg for arg in args if arg is not kwargs)
        return func(*args, **kwargs)

    return wrapper
