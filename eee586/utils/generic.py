import time
import pickle
from pathlib import Path


def get_time(format: str = "%Y_%m_%d_%H_%M"):
    return time.strftime(format)


def pickle_dump(obj: object, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def picklize(func, path: Path, *args, enforce: bool = False, **kwargs):
    if enforce or not path.exists():
        result = func(*args, **kwargs)
        pickle_dump(result, path)
    else:
        result = pickle_load(path)
    return result


def batch_iterable(iterable, batch_size=1):
    l = len(iterable)
    for i in range(0, l, batch_size):
        yield iterable[i : min(i + batch_size, l)]
