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


def test_a_request_refused_before_it_had_a_place_never_queued():
    clock, wall = Clock(), Clock()
    queueing = Queueing(clock=clock, wall=wall)
    held = queueing.arrive(1)
    malformed = queueing.arrive(1)
    clock.now = 0.01
    queueing.leave(malformed, refused=True)
    assert queueing.last_day() == (0, 0)
    queueing.leave(held)
    assert queueing.in_flight == 0


def test_a_request_refused_after_it_had_a_place_still_waited():
    clock, wall = Clock(), Clock()
    queueing = Queueing(clock=clock, wall=wall)
    held = queueing.arrive(1)
    late = queueing.arrive(1)
    clock.now = 2.0
    queueing.leave(held)
    queueing.leave(late, refused=True)
    assert queueing.last_day() == (1, 2000)


def test_a_new_process_holds_nothing_the_last_one_held():
    clock, wall = Clock(), Clock()
    queueing = Queueing(clock=clock, wall=wall)
    stuck = queueing.arrive(1)
    cut_off = queueing.arrive(1)
    clock.now = 5.0
    queueing.restart()
    # The first request to the new process finds its place free.
    fresh = queueing.arrive(1)
    assert not fresh.waiting
    # The old process's requests close later and free nothing of the new one's.
    queueing.leave(stuck)
    queueing.leave(cut_off)
    assert queueing.in_flight == 1
    after = queueing.arrive(1)
    assert after.waiting
    queueing.leave(fresh)
    queueing.leave(after)
    assert queueing.in_flight == 0
    assert queueing.last_day() == (2, 5000)


def test_a_request_knows_how_long_it_waited_for_its_place():
    clock, wall = Clock(), Clock()
    queueing = Queueing(clock=clock, wall=wall)
    held = queueing.arrive(1)
    late = queueing.arrive(1)
    assert queueing.waited_ms(held) == 0
    clock.now = 1.5
    assert queueing.waited_ms(late) == 1500
    queueing.leave(held)
    clock.now = 4.0
    assert queueing.waited_ms(late) == 1500
