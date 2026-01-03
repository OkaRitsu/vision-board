from abc import ABC, abstractmethod


class ReducerStrategy(ABC):
    @abstractmethod
    def build(self, **config):
        pass

    @abstractmethod
    def reduce(self, features):
        pass
