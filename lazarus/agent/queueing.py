"""The requests that waited for a place in a language model's engine.

A llama-server holds as many requests at once as it has slots, and mlx-lm
one; a request that arrives with every place taken waits until one is
given back, and the places are given to the waiting requests in the order
they came. The agent is the one way in, so it sees each request arrive and
leave: a request queued when as many were already in flight as there are
places, and waited until a request holding a place left, or until it gave
up itself. One the engine refused before it had a place never waited for
one. Kept by the hour for the last day, beside the deployment rather than
its process, so a relaunch does not forget it; what is in flight is the
process's, and a new process starts with none.
"""

from __future__ import annotations

import time
from collections import deque

HOURS_KEPT = 24


class Ticket:
    """One request through the gate: when it came, whether it is still
    waiting for a place and how long it waited for one, the hour it was
    counted as queued in, and the process it came to."""

    __slots__ = ("arrived", "waiting", "waited_ms", "hour", "process")

    def __init__(self, arrived: float, waiting: bool, process: int) -> None:
        self.arrived = arrived
        self.waiting = waiting
        self.waited_ms = 0
        self.hour: int | None = None
        self.process = process


class Queueing:
    def __init__(self, clock=time.monotonic, wall=time.time) -> None:
        self.clock = clock
        self.wall = wall
        self.process = 0
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
        ticket.waited_ms = int((self.clock() - ticket.arrived) * 1000)
        bucket = self._hour()
        bucket[2] = max(bucket[2], ticket.waited_ms)

    def arrive(self, places: int) -> Ticket:
        """A request arrives at an engine with that many places."""
        ticket = Ticket(self.clock(), self.in_flight >= places, self.process)
        self.in_flight += 1
        if ticket.waiting:
            self.waiting.append(ticket)
            bucket = self._hour()
            bucket[1] += 1
            ticket.hour = bucket[0]
        return ticket

    def leave(self, ticket: Ticket, refused: bool = False) -> None:
        """A request is done: one that held a place gives it to the request
        that has waited longest; one still waiting gave up, or was refused
        and so never queued. One the last process held holds nothing here."""
        if ticket.process != self.process:
            return
        self.in_flight -= 1
        if ticket.waiting:
            self.waiting.remove(ticket)
            if refused:
                ticket.waiting = False
                for bucket in self.hours:
                    if bucket[0] == ticket.hour:
                        bucket[1] -= 1
            else:
                self._waited(ticket)
        elif self.waiting:
            self._waited(self.waiting.popleft())

    def waited_ms(self, ticket: Ticket) -> int:
        """How long the request waited for its place, or has waited so far."""
        return int((self.clock() - ticket.arrived) * 1000) if ticket.waiting else ticket.waited_ms

    def restart(self) -> None:
        """A new process serves the deployment: the requests the last one
        held are not its own, and those waiting for it waited until it went."""
        for ticket in self.waiting:
            self._waited(ticket)
        self.waiting.clear()
        self.in_flight = 0
        self.process += 1

    def last_day(self) -> tuple[int, int]:
        """The requests queued in the last day, and the longest any waited,
        counting those waiting still."""
        self._hour()
        queued = sum(bucket[1] for bucket in self.hours)
        longest = max((bucket[2] for bucket in self.hours), default=0)
        if self.waiting:
            longest = max(longest, int((self.clock() - self.waiting[0].arrived) * 1000))
        return queued, longest
