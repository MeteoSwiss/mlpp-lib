from functools import wraps


def with_attrs(**func_attrs):
    """
    Set attributes in the decorated function, at definition time.
    Only accepts keyword arguments.
    """

    def attr_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        for attr, value in func_attrs.items():
            setattr(wrapper, attr, value)

        return wrapper

    return attr_decorator
