from matplotlib import pyplot as plt

class Logger():
    """
    Class for logging outputs for each component. Usually internal to each
    component.
    """

    def __init__(self, level: int, prefix: str):
        self._level = level
        self._prefix = prefix

    def log(self, level: int, msg: str):
        if level <= self._level:
            print(f"{self._prefix} -- {msg}")

    def plot(self, level: int, fname: str, *plot_args, **plot_kwargs):
        if level <= self._level:
            func = getattr(plt, fname)
            func(*plot_args, **plot_kwargs)

