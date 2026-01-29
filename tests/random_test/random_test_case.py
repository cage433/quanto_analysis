import random
from functools import wraps
from typing import Optional

import numpy as np


__all__ = [
    "RandomisedTest",
    "SeedGenerator",
]

from modelling.maths.random.random_number_generator import RandomNumberGenerator


class SeedGenerator:
    """
    Provides a deterministic seed iterator for random test_utils
    """

    def __init__(self, initial_seed=None):
        self._random = np.random.RandomState(initial_seed or random.randint(0, 100 * 1000 * 1000))

    def __next__(self):
        return self._random.randint(0, 100 * 1000 * 1000)


class RandomisedTest:  # pragma: no cover
    """ Context manager around a test_utils case that has a saltable rng

        seed_generator is provided when running a deterministic set of 'randomised' test_utils.
        set 'seed' to reproduce a specific failing test_utils
    """

    def __init__(self, seed=None, number_of_runs=1, seed_gen=SeedGenerator(), verbose=False, num_allowed_failures=0):
        self.original_seed: Optional[int] = seed
        self.seed_gen: SeedGenerator = seed_gen
        self.current_seed: int = None
        self.rng: Optional[RandomNumberGenerator] = None
        self.number_of_runs = number_of_runs if seed is None else 1
        self.verbose: bool = verbose
        self.num_allowed_failures: int = num_allowed_failures

    def msg(self, text):
        return f"Seed {self.current_seed}, {text}"

    def __call__(self, func):

        @wraps(func)
        def inner(*args):
            num_failures = 0
            failing_seeds = []
            for i in range(1, self.number_of_runs + 1):
                self.current_seed = self.original_seed or next(self.seed_gen)
                self.rng: RandomNumberGenerator = RandomNumberGenerator(self.current_seed)
                if self.verbose:
                    print(f"Running {func.__name__} iteration {i} of {self.number_of_runs} with seed {self.current_seed}")
                try:
                    func(*args, self.rng)
                except AssertionError:
                    num_failures += 1
                    failing_seeds.append(self.current_seed)
                    if num_failures > self.num_allowed_failures or self.original_seed:
                        if self.num_allowed_failures > 0 and self.original_seed is None:
                            seeds_text = ", ".join(map(str, failing_seeds))
                            print(
                                f"\nTest '{func.__name__}' had more than {self.num_allowed_failures},"
                                f" after {i} attempts. Failing seeds were  [{seeds_text}]"
                            )
                        else:
                            print(f"\nTest '{func.__name__}' failed with seed {self.current_seed}")
                        raise
                except Exception:
                    print(f"\nTest '{func.__name__}' threw an exception with seed {self.current_seed}")
                    raise

        return inner
