"""
fsrs.models
-----------

This module defines the core classes used by the FSRS scheduler.

Classes:
    State: Enum representing the learning state of a Card object.
    Rating: Enum representing the four possible ratings when reviewing a card.
    ReviewLog: Represents the log entry of Card that has been reviewed.
    Card: Represents a flashcard in the FSRS system.
    SchedulingInfo: Simple data class that bundles together an updated Card object and it's corresponding ReviewLog object.
    SchedulingCards: Manages the scheduling of a Card object for each of the four potential ratings.
    Parameters: The parameters used to configure the FSRS scheduler.
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Optional, Union
from enum import IntEnum


class State(IntEnum):
    """
    Enum representing the learning state of a Card object.
    """

    New = 0
    Learning = 1
    Review = 2
    Relearning = 3


class Rating(IntEnum):
    """
    Enum representing the four possible ratings when reviewing a card.
    """

    Again = 1
    Hard = 2
    Good = 3
    Easy = 4


class ReviewLog:
    """
    Represents the log entry of Card that has been reviewed.

    Attributes:
        rating (Rating): The rating given to the card during the review.
        scheduled_days (int): The number of days until the card is due next.
        elapsed_days (int): The number of days since the card was last reviewed.
        review (datetime): The date and time of the review.
        state (State): The learning state of the card before the review.
    """

    rating: Rating
    scheduled_days: int
    elapsed_days: int
    review: datetime
    state: State

    def __init__(
        self,
        rating: Rating,
        scheduled_days: int,
        elapsed_days: int,
        review: datetime,
        state: State,
    ) -> None:
        """
        Creates and initializes a ReviewLog object.

        Args:
            rating (Rating): The rating given to the card during the review.
            scheduled_days (int): The number of days until the card is due next.
            elapsed_days (int): The number of days since the card was last reviewed.
            review (datetime): The date and time of the review.
            state (State): The learning state of the card before the review.
        """
        self.rating = rating
        self.scheduled_days = scheduled_days
        self.elapsed_days = elapsed_days
        self.review = review
        self.state = state

    def to_dict(self) -> dict[str, Union[int, str]]:
        """
        Returns a JSON-serializable dictionary representation of the ReviewLog object.

        This method is specifically useful for storing ReviewLog objects in a database.

        Returns:
            dict: A dictionary representation of the ReviewLog object.
        """
        return_dict = {
            "rating": self.rating.value,
            "scheduled_days": self.scheduled_days,
            "elapsed_days": self.elapsed_days,
            "review": self.review.isoformat(),
            "state": self.state.value,
        }

        return return_dict

    @staticmethod
    def from_dict(source_dict: dict[str, Any]) -> "ReviewLog":
        """
        Creates a ReviewLog object from an existing dictionary.

        Args:
            source_dict (dict[str, Any]): A dictionary representing an existing ReviewLog object.

        Returns:
            ReviewLog: A ReviewLog object created from the provided dictionary.
        """
        rating = Rating(int(source_dict["rating"]))
        scheduled_days = int(source_dict["scheduled_days"])
        elapsed_days = int(source_dict["elapsed_days"])
        review = datetime.fromisoformat(source_dict["review"])
        state = State(int(source_dict["state"]))

        return ReviewLog(
            rating,
            scheduled_days,
            elapsed_days,
            review,
            state,
        )


class Card:
    """
    Represents a flashcard in the FSRS system.

    Attributes:
        due (datetime): The date and time when the card is due next.
        stability (float): Core FSRS parameter used for scheduling.
        difficulty (float): Core FSRS parameter used for scheduling.
        elapsed_days (int): The number of days since the card was last reviewed.
        scheduled_days (int): The number of days until the card is due next.
        reps (int): The number of times the card has been reviewed in its history.
        lapses (int): The number of times the card has been lapsed in its history.
        state (State): The card's current learning state.
        last_review (datetime): The date and time of the card's last review.
    """

    due: datetime
    stability: float
    difficulty: float
    elapsed_days: int
    scheduled_days: int
    reps: int
    lapses: int
    state: State
    last_review: Optional[datetime]

    def __init__(
        self,
        due: Optional[datetime] = None,
        stability: float = 0,
        difficulty: float = 0,
        elapsed_days: int = 0,
        scheduled_days: int = 0,
        reps: int = 0,
        lapses: int = 0,
        state: State = State.New,
        last_review: Optional[datetime] = None,
    ) -> None:
        """
        Creates and initializes a Card object.

        Note that each of the arguments for this method are optional and can be omitted when creating a new Card.

        Args:
            due (Optional[datetime]): The date and time when the card is due next.
            stability (float): Core FSRS parameter used for scheduling.
            difficulty (float): Core FSRS parameter used for scheduling.
            elapsed_days (int): The number of days since the card was last reviewed.
            scheduled_days (int): The number of days until the card is due next.
            reps (int): The number of times the card has been reviewed in its history.
            lapses (int): The number of times the card has been lapsed in its history.
            state (State): The card's current learning state.
            last_review (Optional[datetime]): The date and time of the card's last review.
        """
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
        self.last_review = last_review

    def to_dict(self) -> dict[str, Any]:
        """
        Returns a JSON-serializable dictionary representation of the Card object.

        This method is specifically useful for storing Card objects in a database.

        Returns:
            dict: A dictionary representation of the Card object.
        """
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
            return_dict["last_review"] = (
                self.last_review.isoformat() if self.last_review else None
            )

        return return_dict

    @staticmethod
    def from_dict(source_dict: dict[str, Any]) -> "Card":
        """
        Creates a Card object from an existing dictionary.

        Args:
            source_dict (dict[str, Any]): A dictionary representing an existing Card object.

        Returns:
            ReviewLog: A Card object created from the provided dictionary.
        """
        due = datetime.fromisoformat(source_dict["due"])
        stability = float(source_dict["stability"])
        difficulty = float(source_dict["difficulty"])
        elapsed_days = int(source_dict["elapsed_days"])
        scheduled_days = int(source_dict["scheduled_days"])
        reps = int(source_dict["reps"])
        lapses = int(source_dict["lapses"])
        state = State(int(source_dict["state"]))

        if "last_review" in source_dict and source_dict["last_review"]:
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

    def get_retrievability(self, now: Optional[datetime] = None) -> float:
        """
        Calculates the Card object's current retrievability for a given date and time.

        Args:
            now (datetime): The current date and time

        Returns:
            float: The retrievability of the Card object.
        """
        DECAY = -0.5
        FACTOR = 0.9 ** (1 / DECAY) - 1

        if now is None:
            now = datetime.now(timezone.utc)

        if (
            self.state in (State.Learning, State.Review, State.Relearning)
            and self.last_review
        ):
            elapsed_days = max(0, (now - self.last_review).days)
            return (1 + FACTOR * elapsed_days / self.stability) ** DECAY
        else:
            return 0


@dataclass
class SchedulingInfo:
    """
    Simple data class that bundles together an updated Card object and it's corresponding ReviewLog object.

    This class is specifically used to provide an updated card and it's review log after a card has been reviewed.
    """

    card: Card
    review_log: ReviewLog


class SchedulingCards:
    """
    Manages the scheduling of a Card object for each of the four potential ratings.

    A SchedulingCards object is created from an existing card and creates four new potential cards which
    are updated according to whether the card will be chosen to be reviewed as Again, Hard, Good or Easy.

    Attributes:
        again (Card): An updated Card object that was rated Again.
        hard (Card): An updated Card object that was rated Hard.
        good (Card): An updated Card object that was rated Good.
        easy (Card): An updated Card object that was rated Easy.
    """

    again: Card
    hard: Card
    good: Card
    easy: Card

    def __init__(self, card: Card) -> None:
        self.again = deepcopy(card)
        self.hard = deepcopy(card)
        self.good = deepcopy(card)
        self.easy = deepcopy(card)

    def update_state(self, state: State) -> None:
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
    ) -> None:
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

    def record_log(self, card: Card, now: datetime) -> dict[Rating, SchedulingInfo]:
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
    """
    The parameters used to configure the FSRS scheduler.

    Attributes:
        request_retention (float): The desired retention of the scheduler. Corresponds to the maximum retrievability a Card object can have before it is due.
        maximum_interval (int): The maximum number of days into the future a Card object can be scheduled for next review.
        w (tuple[float, ...]): The 19 model weights of the FSRS scheduler.
    """

    request_retention: float
    maximum_interval: int
    w: tuple[float, ...]
    enable_short_term: bool
    DECAY: float = -0.5
    FACTOR: float = 0.9 ** (1 / DECAY) - 1

    def __init__(
        self,
        w: Optional[tuple[float, ...]] = None,
        request_retention: Optional[float] = None,
        maximum_interval: Optional[int] = None,
    ) -> None:
        self.w = (
            w
            if w is not None
            else (
                0.4072,
                1.1829,
                3.1262,
                15.4722,
                7.2102,
                0.5316,
                1.0651,
                0.0234,
                1.616,
                0.1544,
                1.0824,
                1.9813,
                0.0953,
                0.2975,
                2.2042,
                0.2407,
                2.9466,
                0.5034,
                0.6567,
            )
        )
        self.request_retention = (
            request_retention if request_retention is not None else 0.9
        )
        self.maximum_interval = (
            maximum_interval if maximum_interval is not None else 36500
        )
        self.enable_short_term = True

    def init_stability(self, r: Rating) -> float:
        return max(self.w[r - 1], 0.1)

    def init_difficulty(self, r: Rating) -> float:
        # compute initial difficulty and clamp it between 1 and 10
        return min(max(self.w[4] - math.exp(self.w[5] * (r - 1)) + 1, 1), 10)

    def forgetting_curve(self, elapsed_days: int, stability: float) -> float:
        return (1 + self.FACTOR * elapsed_days / stability) ** self.DECAY

    def next_interval(self, s: float, elapsed_day: float) -> int:
        new_interval = (
            s / self.FACTOR * (self.request_retention ** (1 / self.DECAY) - 1)
        )
        return min(max(round(new_interval), 1), self.maximum_interval)

    def next_difficulty(self, d: float, r: Rating) -> float:
        next_d = d - self.w[6] * (r - 3)

        return min(
            max(self.mean_reversion(self.init_difficulty(Rating.Easy), next_d), 1), 10
        )

    def next_short_term_stability(self, stability: float, rating: Rating) -> float:
        return stability * math.exp(self.w[17] * (rating - 3 + self.w[18]))

    def mean_reversion(self, init: float, current: float) -> float:
        return self.w[7] * init + (1 - self.w[7]) * current

    def next_recall_stability(
        self, d: float, s: float, r: float, rating: Rating
    ) -> float:
        hard_penalty = self.w[15] if rating == Rating.Hard else 1
        easy_bonus = self.w[16] if rating == Rating.Easy else 1
        return s * (
            1
            + math.exp(self.w[8])
            * (11 - d)
            * math.pow(s, -self.w[9])
            * (math.exp((1 - r) * self.w[10]) - 1)
            * hard_penalty
            * easy_bonus
        )

    def next_forget_stability(self, d: float, s: float, r: float) -> float:
        return (
            self.w[11]
            * math.pow(d, -self.w[12])
            * (math.pow(s + 1, self.w[13]) - 1)
            * math.exp((1 - r) * self.w[14])
        )
