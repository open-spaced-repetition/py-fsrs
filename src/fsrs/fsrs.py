"""
fsrs.fsrs
---------

This module defines each of the classes used in the fsrs package.

Classes:
    State: Enum representing the learning state of a Card object.
    Rating: Enum representing the four possible ratings when reviewing a card.
    ReviewLog: Represents the log entry of Card that has been reviewed.
    Card: Represents a flashcard in the FSRS system.
    FSRS: The FSRS scheduler.
    SchedulingInfo: Simple data class that bundles together an updated Card object and it's corresponding ReviewLog object.
    SchedulingCards: Manages the scheduling of a Card object for each of the four potential ratings.
    Parameters: The parameters used to configure the FSRS scheduler.
"""

import math
from datetime import datetime, timezone, timedelta
import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
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
    last_review: datetime

    def __init__(
        self,
        due: datetime | None = None,
        stability: float = 0,
        difficulty: float = 0,
        elapsed_days: int = 0,
        scheduled_days: int = 0,
        reps: int = 0,
        lapses: int = 0,
        state: State = State.New,
        last_review: datetime | None = None,
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

        if last_review is not None:
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
            return_dict["last_review"] = self.last_review.isoformat()

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

    def get_retrievability(self, now: datetime | None = None) -> float:
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

        if self.state in (State.Learning, State.Review, State.Relearning):
            elapsed_days = max(0, (now - self.last_review).days)
            return (1 + FACTOR * elapsed_days / self.stability) ** DECAY
        else:
            return 0

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

    #def to_dict(self) -> dict[str, Union[int, str]]:
    def to_dict(self) -> dict[str, int | str]:
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
        self.again = copy.deepcopy(card)
        self.hard = copy.deepcopy(card)
        self.good = copy.deepcopy(card)
        self.easy = copy.deepcopy(card)

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

    def __init__(
        self,
        w: tuple[float, ...] | None = None,
        request_retention: float | None = None,
        maximum_interval: int | None = None,
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

class FSRS:
    """
    The FSRS scheduler.

    Enables the reviewing and future scheduling of cards according to the FSRS algorithm.

    Attributes:
        p (Parameters): Object for configuring the scheduler's model weights, desired retention and maximum interval.
        DECAY (float): Constant used to model the forgetting curve and compute the length of a Card's next interval after being repeated.
        FACTOR (float): Constant used to model the forgetting curve and compute the length of a Card's next interval after being repeated.
    """

    p: Parameters
    DECAY: float
    FACTOR: float

    def __init__(
        self,
        w: tuple[float, ...] | None = None,
        request_retention: float | None = None,
        maximum_interval: int | None = None,
    ) -> None:
        """
        Initializes the FSRS scheduler.

        Args:
            w (Optional[tuple[float, ...]]): The 19 model weights of the FSRS scheduler.
            request_retention (Optional[float]): The desired retention of the scheduler. Corresponds to the maximum retrievability a Card object can have before it is due.
            maximum_interval (Optional[int]): The maximum number of days into the future a Card object can be scheduled for next review.
        """
        self.p = Parameters(w, request_retention, maximum_interval)
        self.DECAY = -0.5
        self.FACTOR = 0.9 ** (1 / self.DECAY) - 1

    def review_card(
        self, card: Card, rating: Rating, now: datetime | None = None
    ) -> tuple[Card, ReviewLog]:
        """
        Reviews a card for a given rating.

        Args:
            card (Card): The card being reviewed.
            rating (Rating): The chosen rating for the card being reviewed.
            now (Optional[datetime]): The date and time of the review.

        Returns:
            tuple: A tuple containing the updated, reviewed card and its corresponding review log.

        Raises:
            ValueError: If the `now` argument is not timezone-aware and set to UTC.
        """
        scheduling_cards = self.repeat(card, now)

        card = scheduling_cards[rating].card
        review_log = scheduling_cards[rating].review_log

        return card, review_log

    def repeat(
        self, card: Card, now: datetime | None = None
    ) -> dict[Rating, SchedulingInfo]:
        if now is None:
            now = datetime.now(timezone.utc)

        if (now.tzinfo is None) or (now.tzinfo != timezone.utc):
            raise ValueError("datetime must be timezone-aware and set to UTC")

        card = copy.deepcopy(card)
        if card.state == State.New:
            card.elapsed_days = 0
        else:
            card.elapsed_days = (now - card.last_review).days
        card.last_review = now
        card.reps += 1
        s = SchedulingCards(card)
        s.update_state(card.state)

        if card.state == State.New:
            self.init_ds(s)

            s.again.due = now + timedelta(minutes=1)
            s.hard.due = now + timedelta(minutes=5)
            s.good.due = now + timedelta(minutes=10)
            easy_interval = self.next_interval(s.easy.stability)
            s.easy.scheduled_days = easy_interval
            s.easy.due = now + timedelta(days=easy_interval)
        elif card.state == State.Learning or card.state == State.Relearning:
            interval = card.elapsed_days
            last_d = card.difficulty
            last_s = card.stability
            retrievability = self.forgetting_curve(interval, last_s)
            self.next_ds(s, last_d, last_s, retrievability, card.state)

            hard_interval = 0
            good_interval = self.next_interval(s.good.stability)
            easy_interval = max(self.next_interval(s.easy.stability), good_interval + 1)
            s.schedule(now, hard_interval, good_interval, easy_interval)
        elif card.state == State.Review:
            interval = card.elapsed_days
            last_d = card.difficulty
            last_s = card.stability
            retrievability = self.forgetting_curve(interval, last_s)
            self.next_ds(s, last_d, last_s, retrievability, card.state)

            hard_interval = self.next_interval(s.hard.stability)
            good_interval = self.next_interval(s.good.stability)
            hard_interval = min(hard_interval, good_interval)
            good_interval = max(good_interval, hard_interval + 1)
            easy_interval = max(self.next_interval(s.easy.stability), good_interval + 1)
            s.schedule(now, hard_interval, good_interval, easy_interval)
        return s.record_log(card, now)

    def init_ds(self, s: SchedulingCards) -> None:
        s.again.difficulty = self.init_difficulty(Rating.Again)
        s.again.stability = self.init_stability(Rating.Again)
        s.hard.difficulty = self.init_difficulty(Rating.Hard)
        s.hard.stability = self.init_stability(Rating.Hard)
        s.good.difficulty = self.init_difficulty(Rating.Good)
        s.good.stability = self.init_stability(Rating.Good)
        s.easy.difficulty = self.init_difficulty(Rating.Easy)
        s.easy.stability = self.init_stability(Rating.Easy)

    def next_ds(
        self,
        s: SchedulingCards,
        last_d: float,
        last_s: float,
        retrievability: float,
        state: State,
    ) -> None:
        s.again.difficulty = self.next_difficulty(last_d, Rating.Again)
        s.hard.difficulty = self.next_difficulty(last_d, Rating.Hard)
        s.good.difficulty = self.next_difficulty(last_d, Rating.Good)
        s.easy.difficulty = self.next_difficulty(last_d, Rating.Easy)

        if state == State.Learning or state == State.Relearning:
            # compute short term stabilities
            s.again.stability = self.short_term_stability(last_s, Rating.Again)
            s.hard.stability = self.short_term_stability(last_s, Rating.Hard)
            s.good.stability = self.short_term_stability(last_s, Rating.Good)
            s.easy.stability = self.short_term_stability(last_s, Rating.Easy)

        elif state == State.Review:
            s.again.stability = self.next_forget_stability(
                last_d, last_s, retrievability
            )
            s.hard.stability = self.next_recall_stability(
                last_d, last_s, retrievability, Rating.Hard
            )
            s.good.stability = self.next_recall_stability(
                last_d, last_s, retrievability, Rating.Good
            )
            s.easy.stability = self.next_recall_stability(
                last_d, last_s, retrievability, Rating.Easy
            )

    def init_stability(self, r: Rating) -> float:
        return max(self.p.w[r - 1], 0.1)

    def init_difficulty(self, r: Rating) -> float:
        # compute initial difficulty and clamp it between 1 and 10
        return min(max(self.p.w[4] - math.exp(self.p.w[5] * (r - 1)) + 1, 1), 10)

    def forgetting_curve(self, elapsed_days: int, stability: float) -> float:
        return (1 + self.FACTOR * elapsed_days / stability) ** self.DECAY

    def next_interval(self, s: float) -> int:
        new_interval = (
            s / self.FACTOR * (self.p.request_retention ** (1 / self.DECAY) - 1)
        )
        return min(max(round(new_interval), 1), self.p.maximum_interval)

    def next_difficulty(self, d: float, r: Rating) -> float:
        next_d = d - self.p.w[6] * (r - 3)

        return min(
            max(self.mean_reversion(self.init_difficulty(Rating.Easy), next_d), 1), 10
        )

    def short_term_stability(self, stability: float, rating: Rating) -> float:
        return stability * math.exp(self.p.w[17] * (rating - 3 + self.p.w[18]))

    def mean_reversion(self, init: float, current: float) -> float:
        return self.p.w[7] * init + (1 - self.p.w[7]) * current

    def next_recall_stability(
        self, d: float, s: float, r: float, rating: Rating
    ) -> float:
        hard_penalty = self.p.w[15] if rating == Rating.Hard else 1
        easy_bonus = self.p.w[16] if rating == Rating.Easy else 1
        return s * (
            1
            + math.exp(self.p.w[8])
            * (11 - d)
            * math.pow(s, -self.p.w[9])
            * (math.exp((1 - r) * self.p.w[10]) - 1)
            * hard_penalty
            * easy_bonus
        )

    def next_forget_stability(self, d: float, s: float, r: float) -> float:
        return (
            self.p.w[11]
            * math.pow(d, -self.p.w[12])
            * (math.pow(s + 1, self.p.w[13]) - 1)
            * math.exp((1 - r) * self.p.w[14])
        )
