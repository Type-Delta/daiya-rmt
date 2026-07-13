from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future
from typing import TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def bounded_ordered_map(
    executor: Executor,
    function: Callable[[InputT], OutputT],
    items: Iterable[InputT],
    max_in_flight: int,
) -> Iterator[OutputT]:
    """Submit at most ``max_in_flight`` synchronous jobs and yield in input order."""
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be at least 1")

    pending: deque[Future[OutputT]] = deque()
    iterator = iter(items)

    def fill() -> None:
        while len(pending) < max_in_flight:
            try:
                item = next(iterator)
            except StopIteration:
                return
            pending.append(executor.submit(function, item))

    fill()
    while pending:
        # Waiting for the oldest submitted job gives deterministic output while
        # later jobs can still run in the bounded window.
        yield pending.popleft().result()
        fill()
