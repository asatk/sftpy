import abc

class Actor(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def simulate(self) -> None:
        ...

    def __call__(self):
        self.simulate()