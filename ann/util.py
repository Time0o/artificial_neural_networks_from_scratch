def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def make_iterable(obj):
    return [obj] if not is_iterable(obj) else obj
