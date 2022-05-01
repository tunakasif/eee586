import time


def get_time(format: str = "%Y_%m_%d_%H_%M"):
    return time.strftime(format)
