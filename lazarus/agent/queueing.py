"""The requests that waited for a place in a language model's engine.

A llama-server holds as many requests at once as it has slots, and mlx-lm
one; a request that arrives with every place taken waits until one is
given back, and the places are given to the waiting requests in the order
they came. The agent is the one way in, so it sees each request arrive and
leave: a request queued when as many were already in flight as there are
places, and waited until a request holding a place left, or until it gave
up itself. Kept by the hour for the last day, beside the deployment rather
than its process, so a relaunch does not forget it.
"""

from __future__ import annotations

import time
from collections import deque

HOURS_KEPT = 24


class Ticket:
    """One request through the gate: when it came, and whether it is still
    waiting for a place."""

    __slots__ = ("arrived", "waiting")

    def __init__(self, arrived: float, waiting: bool) -> None:
        self.arrived = arrived
        self.waiting = waiting


class Queueing:
    def __init__(self, clock=time.monotonic, wall=time.time) -> None:
        self.clock = clock
        self.wall = wall
        self.in_flight = 0
        self.waiting: deque[Ticket] = deque()
        # (hour, requests queued, longest wait in milliseconds)
        self.hours: deque[list[int]] = deque()

    def _hour(self) -> list[int]:
        hour = int(self.wall() // 3600)
        if not self.hours or self.hours[-1][0] != hour:
            self.hours.append([hour, 0, 0])
        while self.hours and self.hours[0][0] <= hour - HOURS_KEPT:
            self.hours.popleft()
        return self.hours[-1]

    def _waited(self, ticket: Ticket) -> None:
        ticket.waiting = False
        bucket = self._hour()
        bucket[2] = max(bucket[2], int((self.clock() - ticket.arrived) * 1000))

    def arrive(self, places: int) -> Ticket:
        """A request arrives at an engine with that many places."""
        ticket = Ticket(self.clock(), self.in_flight >= places)
        self.in_flight += 1
        if ticket.waiting:
            self.waiting.append(ticket)
            self._hour()[1] += 1
        return ticket

    def leave(self, ticket: Ticket) -> None:
        """A request is done: one that held a place gives it to the request
        that has waited longest; one still waiting gave up."""
        self.in_flight -= 1
        if ticket.waiting:
            self.waiting.remove(ticket)
            self._waited(ticket)
        elif self.waiting:
            self._waited(self.waiting.popleft())

    def last_day(self) -> tuple[int, int]:
        """The requests queued in the last day, and the longest any waited,
        counting those waiting still."""
        self._hour()
        queued = sum(bucket[1] for bucket in self.hours)
        longest = max((bucket[2] for bucket in self.hours), default=0)
        if self.waiting:
            longest = max(longest, int((self.clock() - self.waiting[0].arrived) * 1000))
        return queued, longest
