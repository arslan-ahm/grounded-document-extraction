"""Seeding and CPU thread limits.

Two separate concerns live here because they have the same failure mode: if
either is forgotten, results stop being reproducible and the repository's
central promise is void.

*Seeding* covers Python's ``random``, NumPy and torch. *Thread limits* matter
because this project is developed on a 4-core machine shared with other jobs;
an unbounded ``torch.set_num_threads`` both slows the machine down and makes
latency measurements meaningless. The limit is applied in every entry point,
not only in ``__main__``, so importing the package from a notebook behaves the
same way as running a script.
"""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterator

import numpy as np
import torch

DEFAULT_THREADS = 2


def limit_threads(n_threads: int = DEFAULT_THREADS) -> None:
    """Cap CPU parallelism for torch and the BLAS backends.

    The environment variables are set as well as ``torch.set_num_threads``
    because BLAS libraries read them at their own first use, which may happen
    inside a dependency we do not control.

    Args:
        n_threads: Maximum threads. Values below 1 are clamped to 1.
    """
    n = max(1, int(n_threads))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    torch.set_num_threads(n)
    with contextlib.suppress(AttributeError, RuntimeError):
        torch.set_num_interop_threads(1)


def seed_everything(seed: int, n_threads: int = DEFAULT_THREADS) -> torch.Generator:
    """Seed every RNG this project touches and return a torch generator.

    Args:
        seed: The seed. Must be a non-negative integer.
        n_threads: Passed to :func:`limit_threads`.

    Returns:
        A ``torch.Generator`` seeded with ``seed``, for callers that want an
        explicit generator rather than the global one (``DataLoader``, dropout
        in a functional call site).

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    limit_threads(n_threads)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


@contextlib.contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Run a block under ``seed`` and restore the previous RNG states after.

    Used where a deterministic sub-computation (a bootstrap, a data probe) must
    not perturb the stream that the surrounding training loop is drawing from.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def worker_seed(base_seed: int, worker_id: int) -> int:
    """Derive a distinct, reproducible seed for a data-loading worker.

    A plain ``base_seed + worker_id`` collides across configurations that differ
    only in worker count, so the two are mixed instead.
    """
    return (base_seed * 1_000_003 + worker_id * 7919) % (2**31 - 1)
