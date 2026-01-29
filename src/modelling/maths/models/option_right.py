__all__ = [
    "OptionRight",
    "CALL", "PUT"
]

from abc import ABC, abstractmethod
from numbers import Number


class OptionRight(ABC):
    @abstractmethod
    def intrinsic(self, F: Number, K: Number) -> float:
        raise ValueError("Not implemented")

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return type(self).__hash__()

    def __repr__(self):
        return str(self)


class _Call(OptionRight):
    def intrinsic(self, F: Number, K: Number) -> float:
        return max(F - K, 0)

    def __str__(self):
        return "Call"

class _Put(OptionRight):
    def intrinsic(self, F: Number, K: Number) -> float:
        return max(K - F, 0)

    def __str__(self):
        return "Put"

CALL = _Call()
PUT = _Put()
