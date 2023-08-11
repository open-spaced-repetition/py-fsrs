from datetime import datetime
from typing import Tuple
from enum import IntEnum


class State(IntEnum):
    New = 0
    Learning = 1
    Review = 2
    Relearning = 3


class Rating(IntEnum):
    Again = 1
    Hard = 2
    Good = 3
    Easy = 4


class ReviewLog:
    rating: int
    elapsed_days: int
    scheduled_days: int
    Review: datetime
    state: int

    def __init__(
        self,
        rating: int,
        elapsed_days: int,
        scheduled_days: int,
        review: datetime,
        state: int,
    ):
        self.rating = rating
        self.elapsed_days = elapsed_days
        self.scheduled_days = scheduled_days
        self.review = review
        self.state = state


class Card:
    due: datetime
    stability: float
    difficulty: float
    elapsed_days: int
    scheduled_days: int
    reps: int
    lapses: int
    state: State
    last_review: datetime

    def __init__(self) -> None:
        self.due = datetime.utcnow()
        self.stability = 0
        self.difficulty = 0
        self.elapsed_days = 0
        self.scheduled_days = 0
        self.reps = 0
        self.lapses = 0
        self.state = State.New

    def get_retrievability(self) -> float:
        return (1 + self.elapsed_days / (9 * self.stability)) ** -1

    def save_log(
        self,
        rating: Rating,
    ) -> ReviewLog:
        return ReviewLog(
            rating,
            self.elapsed_days,
            self.scheduled_days,
            datetime.utcnow(),
            self.state,
        )

    def update_state(self, rating: Rating):
        match (self.state):
            case State.New:
                if rating == Rating.Again:
                    self.lapses += 1

                if rating == Rating.Easy:
                    self.state = State.Review
                else:
                    self.state = State.Learning

            case State.Learning | State.Relearning:
                if rating == Rating.Good or rating == Rating.Easy:
                    self.state = State.Review

            case State.Review:
                if rating == Rating.Again:
                    self.lapses += 1
                    self.state = State.Relearning


class Parameters:
    request_retention: float
    maximum_interval: int
    w: Tuple[float, ...]

    def __init__(self) -> None:
        self.request_retention = 0.9
        self.maximum_interval = 36500
        self.w = (
            0.4,
            0.6,
            2.4,
            5.8,
            4.93,
            0.94,
            0.86,
            0.01,
            1.49,
            0.14,
            0.94,
            2.18,
            0.05,
            0.34,
            1.26,
            0.29,
            2.61,
        )
