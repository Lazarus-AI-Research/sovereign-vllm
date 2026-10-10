from lazarus.agent.queueing import Queueing


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_a_request_that_finds_every_place_taken_waits_until_one_is_given_back():
    clock, wall = Clock(), Clock()
    queueing = Queueing(clock=clock, wall=wall)
    held = [queueing.arrive(2), queueing.arrive(2)]
    assert queueing.last_day() == (0, 0)
    late = queueing.arrive(2)
    clock.now = 3.0
    # Waiting still: counted, and its wait so far is the longest.
    assert queueing.last_day() == (1, 3000)
    queueing.leave(held[0])
    clock.now = 10.0
    queueing.leave(late)
    queueing.leave(held[1])
    assert queueing.last_day() == (1, 3000)


def test_places_go_to_the_requests_in_the_order_they_came():
    clock, wall = Clock(), Clock()
    queueing = Queueing(clock=clock, wall=wall)
    held = queueing.arrive(1)
    first = queueing.arrive(1)
    clock.now = 1.0
    second = queueing.arrive(1)
    clock.now = 4.0
    queueing.leave(held)
    assert not first.waiting and second.waiting
    clock.now = 9.0
    queueing.leave(first)
    queueing.leave(second)
    assert queueing.last_day() == (2, 8000)


def test_a_request_that_gives_up_while_waiting_frees_no_place():
    clock, wall = Clock(), Clock()
    queueing = Queueing(clock=clock, wall=wall)
    held = queueing.arrive(1)
    patient, impatient = queueing.arrive(1), queueing.arrive(1)
    clock.now = 2.0
    queueing.leave(impatient)
    assert patient.waiting and held.waiting is False
    queueing.leave(held)
    assert not patient.waiting


def test_a_day_later_the_queue_is_forgotten():
    clock, wall = Clock(), Clock()
    queueing = Queueing(clock=clock, wall=wall)
    held = queueing.arrive(1)
    queueing.leave(queueing.arrive(1))
    queueing.leave(held)
    assert queueing.last_day()[0] == 1
    wall.now = 25 * 3600
    assert queueing.last_day() == (0, 0)
