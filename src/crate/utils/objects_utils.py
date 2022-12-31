from pathlib import Path

import numpy as np


def deep_copy(target_object):
    if isinstance(target_object, (str, int, float)):
        return target_object
    elif isinstance(target_object, np.ndarray):
        return target_object.copy()
    elif isinstance(target_object, (list, tuple)):
        return [deep_copy(item) for item in target_object.copy()]
    elif isinstance(target_object, dict):
        return {key: deep_copy(value) for key, value in target_object.copy().items()}
    else:
        for key, value in vars(target_object).items():
            setattr(target_object, key, deep_copy(value))
        return target_object


def deep_dictify(target_object):
    if isinstance(target_object, (str, int, float)):
        return target_object
    elif isinstance(target_object, Path):
        return str(target_object)
    elif isinstance(target_object, np.ndarray):
        return target_object.tolist()
    elif isinstance(target_object, (list, tuple)):
        return [deep_dictify(item) for item in target_object]
    elif isinstance(target_object, dict):
        return {str(key): deep_dictify(value) for key, value in target_object.items()}
    else:
        return {str(key): deep_dictify(value) for key, value in vars(target_object).items()}
