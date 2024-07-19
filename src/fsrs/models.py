from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import copy
from typing import Any, Dict, Tuple, Optional
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
    scheduled_days: int
    elapsed_days: int
    review: datetime
    state: int

    def __init__(
        self,
        rating: int,
        scheduled_days: int,
        elapsed_days: int,
        review: datetime,
        state: int,
    ):
        self.rating = rating
        self.scheduled_days = scheduled_days
        self.elapsed_days = elapsed_days
        self.review = review
        self.state = state

    def to_dict(self):
        return_dict = {
            "rating": self.rating,
            "scheduled_days": self.scheduled_days,
            "elapsed_days": self.elapsed_days,
            "review": self.review.isoformat(),
            "state": self.state,
        }

        return return_dict

    @staticmethod
    def from_dict(source_dict: Dict[str, Any]):
        rating = source_dict["rating"]
        scheduled_days = source_dict["scheduled_days"]
        elapsed_days = source_dict["elapsed_days"]
        review = datetime.fromisoformat(source_dict["review"])
        state = source_dict["state"]

        return ReviewLog(
            rating,
            scheduled_days,
            elapsed_days,
            review,
            state,
        )


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

    def __init__(
        self,
        due=None,
        stability=0,
        difficulty=0,
        elapsed_days: int = 0,
        scheduled_days: int = 0,
        reps=0,
        lapses=0,
        state=State.New,
        last_review=None,
    ) -> None:
        if due is None:
            self.due = datetime.now(timezone.utc)
        else:
            self.due = due

        self.stability = stability
        self.difficulty = difficulty
        self.elapsed_days = elapsed_days
        self.scheduled_days = scheduled_days
        self.reps = reps
        self.lapses = lapses
        self.state = state

        if last_review is not None:
            self.last_review = last_review

    def to_dict(self):
        return_dict = {
            "due": self.due.isoformat(),
            "stability": self.stability,
            "difficulty": self.difficulty,
            "elapsed_days": self.elapsed_days,
            "scheduled_days": self.scheduled_days,
            "reps": self.reps,
            "lapses": self.lapses,
            "state": self.state.value,
        }

        if hasattr(self, "last_review"):
            return_dict["last_review"] = self.last_review.isoformat()

        return return_dict

    @staticmethod
    def from_dict(source_dict: Dict[str, Any]):
        due = datetime.fromisoformat(source_dict["due"])
        stability = source_dict["stability"]
        difficulty = source_dict["difficulty"]
        elapsed_days = source_dict["elapsed_days"]
        scheduled_days = source_dict["scheduled_days"]
        reps = source_dict["reps"]
        lapses = source_dict["lapses"]
        state = State(source_dict["state"])

        if "last_review" in source_dict:
            last_review = datetime.fromisoformat(source_dict["last_review"])
        else:
            last_review = None

        return Card(
            due,
            stability,
            difficulty,
            elapsed_days,
            scheduled_days,
            reps,
            lapses,
            state,
            last_review,
        )

    def get_retrievability(self, now: datetime) -> Optional[float]:
        DECAY = -0.5
        FACTOR = 0.9 ** (1 / DECAY) - 1

        if self.state == State.Review:
            elapsed_days = max(0, (now - self.last_review).days)
            return (1 + FACTOR * elapsed_days / self.stability) ** DECAY
        else:
            return None


@dataclass
class SchedulingInfo:
    card: Card
    review_log: ReviewLog


class SchedulingCards:
    again: Card
    hard: Card
    good: Card
    easy: Card

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
            self.hard.state = state
            self.good.state = State.Review
            self.easy.state = State.Review
        elif state == State.Review:
            self.again.state = State.Relearning
            self.hard.state = State.Review
            self.good.state = State.Review
            self.easy.state = State.Review
            self.again.lapses += 1

    def schedule(
        self,
        now: datetime,
        hard_interval: int,
        good_interval: int,
        easy_interval: int,
    ):
        self.again.scheduled_days = 0
        self.hard.scheduled_days = hard_interval
        self.good.scheduled_days = good_interval
        self.easy.scheduled_days = easy_interval
        self.again.due = now + timedelta(minutes=5)
        if hard_interval > 0:
            self.hard.due = now + timedelta(days=hard_interval)
        else:
            self.hard.due = now + timedelta(minutes=10)
        self.good.due = now + timedelta(days=good_interval)
        self.easy.due = now + timedelta(days=easy_interval)

    def record_log(self, card: Card, now: datetime) -> dict[int, SchedulingInfo]:
        return {
            Rating.Again: SchedulingInfo(
                self.again,
                ReviewLog(
                    Rating.Again,
                    self.again.scheduled_days,
                    card.elapsed_days,
                    now,
                    card.state,
                ),
            ),
            Rating.Hard: SchedulingInfo(
                self.hard,
                ReviewLog(
                    Rating.Hard,
                    self.hard.scheduled_days,
                    card.elapsed_days,
                    now,
                    card.state,
                ),
            ),
            Rating.Good: SchedulingInfo(
                self.good,
                ReviewLog(
                    Rating.Good,
                    self.good.scheduled_days,
                    card.elapsed_days,
                    now,
                    card.state,
                ),
            ),
            Rating.Easy: SchedulingInfo(
                self.easy,
                ReviewLog(
                    Rating.Easy,
                    self.easy.scheduled_days,
                    card.elapsed_days,
                    now,
                    card.state,
                ),
            ),
        }


class Parameters:
    request_retention: float
    maximum_interval: int
    w: Tuple[float, ...]

    def __init__(
        self,
        w: Optional[Tuple[float, ...]] = None,
        request_retention: Optional[float] = None,
        maximum_interval: Optional[int] = None,
    ) -> None:
        self.w = (
            w
            if w is not None
            else (
                0.4872,
                1.4003,
                3.7145,
                13.8206,
                5.1618,
                1.2298,
                0.8975,
                0.031,
                1.6474,
                0.1367,
                1.0461,
                2.1072,
                0.0793,
                0.3246,
                1.587,
                0.2272,
                2.8755,
            )
        )
        self.request_retention = (
            request_retention if request_retention is not None else 0.9
        )
        self.maximum_interval = (
            maximum_interval if maximum_interval is not None else 36500
        )
