import time


def get_time(format: str = "%Y%m%d-%H%M%S"):
    return time.strftime(format)
