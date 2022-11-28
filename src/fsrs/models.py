from datetime import datetime, timedelta
import copy
from enum import Enum
from typing import List, Tuple


def datetime_to_timestamp_nano(t: datetime) -> int:
    return int(datetime.timestamp(t.utcnow()) * 1e9)


class State(Enum):
    New = 0
    Learning = 1
    Review = 2
    Relearning = 3


class Rating(Enum):
    Again = 0
    Hard = 1
    Good = 2
    Easy = 3


class ReviewLog:
    id: int
    card_id: int
    state: State
    rating: Rating
    elapsed_days: int
    scheduled_days: int
    review: datetime
    state: State

    def __init__(self, id: int, card_id: int, rating: Rating, elapsed_days: int, scheduled_days: int, review: datetime,
                 state: State):
        self.id = id
        self.card_id = card_id
        self.rating = rating
        self.elapsed_days = elapsed_days
        self.scheduled_days = scheduled_days
        self.review = review
        self.state = state


class Card:
    id: int
    due: datetime
    stability: float
    difficulty: float
    elapsed_days: int
    scheduled_days: int
    reps: int
    lapses: int
    state: State
    last_review: datetime
    review_logs: List[ReviewLog]

    def __init__(self) -> None:
        self.id = datetime_to_timestamp_nano(datetime.utcnow())
        self.stability = 0
        self.difficulty = 0
        self.elapsed_days = 0
        self.scheduled_days = 0
        self.reps = 0
        self.lapses = 0
        self.state = State.New
        self.review_logs = []


class SchedulingCards:
    def __init__(self, card: Card) -> None:
        self.again = copy.deepcopy(card)
        self.hard = copy.deepcopy(card)
        self.good = copy.deepcopy(card)
        self.easy = copy.deepcopy(card)

    def update_state(self, state: State):
        if state == State.New:
            self.again.state = State.Learning
            self.hard.state = State.Learning
            self.good.state = State.Learning
            self.easy.state = State.Review
        elif state == State.Learning or state == State.Relearning:
            self.again.state = state
            self.hard.state = State.Review
            self.good.state = State.Review
            self.easy.state = State.Review
        elif state == State.Review:
            self.again.state = State.Relearning
            self.hard.state = State.Review
            self.good.state = State.Review
            self.easy.state = State.Review

    def schedule(self, now: datetime, hard_interval: float, good_interval: float, easy_interval: float):
        self.again.scheduled_days = 0
        self.hard.scheduled_days = hard_interval
        self.good.scheduled_days = good_interval
        self.easy.scheduled_days = easy_interval
        self.again.due = now + timedelta(minutes=5)
        self.hard.due = now + timedelta(days=hard_interval)
        self.good.due = now + timedelta(days=good_interval)
        self.easy.due = now + timedelta(days=easy_interval)

    def record_log(self, state: State, now: datetime):
        log_id = datetime_to_timestamp_nano(now)
        self.again.review_logs.append(
            ReviewLog(log_id, self.again.id, Rating.Again, self.again.elapsed_days, self.again.scheduled_days, now,
                      state))
        self.hard.review_logs.append(
            ReviewLog(log_id, self.hard.id, Rating.Hard, self.hard.elapsed_days, self.hard.scheduled_days, now,
                      state))
        self.good.review_logs.append(
            ReviewLog(log_id, self.good.id, Rating.Good, self.good.elapsed_days, self.good.scheduled_days, now,
                      state))
        self.easy.review_logs.append(
            ReviewLog(log_id, self.easy.id, Rating.Easy, self.easy.elapsed_days, self.easy.scheduled_days, now,
                      state))


class Parameters:
    request_retention: float
    maximum_interval: int
    easy_bonus: float
    hard_factor: float
    w: Tuple[float, ...]

    def __init__(self) -> None:
        self.request_retention = 0.9
        self.maximum_interval = 36500
        self.easy_bonus = 1.3
        self.hard_factor = 1.2
        self.w = (1., 1., 5., -0.5, -0.5, 0.2, 1.4, -0.12, 0.8, 2., -0.2, 0.2, 1.)
