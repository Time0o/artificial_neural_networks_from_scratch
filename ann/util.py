def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def make_iterable(obj):
    return [obj] if not is_iterable(obj) else obj


def pop_val(d, k, default):
    if k in d:
        res = d[k]
        d.pop(k)
        return res
    else:
        return default
