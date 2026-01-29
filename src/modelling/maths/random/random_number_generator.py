from collections.abc import Iterable
from copy import copy
from typing import Optional

import numpy as np

__all__ = [
    "RandomNumberGenerator",
]



class RandomNumberGenerator:
    """A wrapper around numpy's `random` with additional methods"""

    def __init__(self, seed):
        self.seed = seed
        self._random = np.random.RandomState(seed)

    def uniform(self, x1=None, x2=None, size=None):
        """Returns a float (or ndarray of floats) in the range [0, 1), [0, x1) or [x1, x2) depending on whether
           0, 1 or 2 values are provided, respectively.
        """
        if x1 is None:
            return self._random.uniform(size=size)
        if x2 is None:
            return self._random.uniform(size=size, high=x1)
        return self._random.uniform(size=size, low=x1, high=x2)

    def standard_normals(self, shape):
        return self._random.normal(size=shape)

    def shuffle(self, a):
        b = copy(a)
        self._random.shuffle(b)
        return b

    def choice(self, *a):
        """Can be called with wither a single list, or else a number of arguments"""
        if len(a) == 1 and isinstance(a[0], Iterable):
            choices = list(a[0])
        else:
            choices = a
        return choices[self._random.randint(len(choices))]

    def with_probability(self, x) -> bool:
        return self._random.uniform(0.0, 1.0) <= x

    def is_heads(self) -> bool:
        return self._random.randint(2) != 0

    def maybe(self, x):
        return self.choice(x, None)

    def normal(self, *args, **kwargs):
        return self._random.normal(*args, **kwargs)

    def randint(self, n1, n2=None) -> int:
        return self._random.randint(n1, n2)

    def random_seed(self):
        return self.randint(10 * 1000 * 1000)

    def random_times(self, n_times: int, t0: Optional[float] = None, T: Optional[float] = None) -> np.ndarray:
        assert n_times > 0, f"invalid n_times {n_times}"

        if T is not None:
            if n_times == 1:
                t0 = t0 or T
            else:
                t0 = t0 or T * self.uniform()
        else:
            t0 = t0 or self.uniform()
        if n_times == 1:
            T = T or t0
        else:
            T = T or t0 + self.uniform()

        if n_times == 1:
            assert t0 == T, f"t0 {t0:1.4f} not equal to T {T:1.4f} for single element times"
        else:
            assert t0 < T, f"t0 {t0:1.4f} not before T {T:1.4f}"

        times = [t0]
        while len(times) < n_times:
            times.append(times[-1] + self.uniform(0.1))
        if n_times > 1:
            l1 = times[-1] - t0
            l2 = T - t0
            times = [t0 + (t - t0) * l2 / l1 for t in times]
        return np.array(times)


