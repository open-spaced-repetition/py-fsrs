"""
fsrs.fsrs
---------

This module defines each of the classes used in the fsrs package.

Classes:
    State: Enum representing the learning state of a Card object.
    Rating: Enum representing the four possible ratings when reviewing a card.
    Card: Represents a flashcard in the FSRS system.
    ReviewLog: Represents the log entry of a Card that has been reviewed.
    Scheduler: The FSRS spaced-repetition scheduler.
"""

from collections import defaultdict
from datetime import datetime, timezone, timedelta
from math import exp, inf, pow
from copy import deepcopy
from typing import Any
from enum import IntEnum
import random

DECAY = -0.5
FACTOR = 0.9 ** (1 / DECAY) - 1

FUZZ_RANGES = [
    {"start": 2.5, "end": 7.0, "factor": 0.15},
    {"start": 7.0, "end": 20.0, "factor": 0.1},
    {"start": 20.0, "end": inf, "factor": 0.05},
]


class State(IntEnum):
    """
    Enum representing the learning state of a Card object.
    """

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
        card_id (int): The id of the card. Defaults to the epoch miliseconds of when the card was created.
        state (State): The card's current learning state.
        step (int | None): The card's current learning or relearning step or None if the card is in the Review state.
        stability (float | None): Core mathematical parameter used for future scheduling.
        difficulty (float | None): Core mathematical parameter used for future scheduling.
        due (datetime): The date and time when the card is due next.
        last_review (datetime | None): The date and time of the card's last review.
    """

    card_id: int
    state: State
    step: int | None
    stability: float | None
    difficulty: float | None
    due: datetime
    last_review: datetime | None

    def __init__(
        self,
        card_id: int | None = None,
        state: State = State.Learning,
        step: int | None = None,
        stability: float | None = None,
        difficulty: float | None = None,
        due: datetime | None = None,
        last_review: datetime | None = None,
    ) -> None:
        if card_id is None:
            # epoch miliseconds of when the card was created
            card_id = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.card_id = card_id

        self.state = state

        if self.state == State.Learning and step is None:
            step = 0
        self.step = step

        self.stability = stability
        self.difficulty = difficulty

        if due is None:
            due = datetime.now(timezone.utc)
        self.due = due

        self.last_review = last_review

    def to_dict(self) -> dict[str, int | float | str | None]:
        """
        Returns a JSON-serializable dictionary representation of the Card object.

        This method is specifically useful for storing Card objects in a database.

        Returns:
            dict: A dictionary representation of the Card object.
        """

        return_dict = {
            "card_id": self.card_id,
            "state": self.state.value,
            "step": self.step,
            "stability": self.stability,
            "difficulty": self.difficulty,
            "due": self.due.isoformat(),
            "last_review": self.last_review.isoformat() if self.last_review else None,
        }

        return return_dict

    @staticmethod
    def from_dict(source_dict: dict[str, Any]) -> "Card":
        """
        Creates a Card object from an existing dictionary.

        Args:
            source_dict (dict[str, Any]): A dictionary representing an existing Card object.

        Returns:
            Card: A Card object created from the provided dictionary.
        """

        card_id = int(source_dict["card_id"])
        state = State(int(source_dict["state"]))
        step = source_dict["step"]
        stability = (
            float(source_dict["stability"]) if source_dict["stability"] else None
        )
        difficulty = (
            float(source_dict["difficulty"]) if source_dict["difficulty"] else None
        )
        due = datetime.fromisoformat(source_dict["due"])
        last_review = (
            datetime.fromisoformat(source_dict["last_review"])
            if source_dict["last_review"]
            else None
        )

        return Card(
            card_id=card_id,
            state=state,
            step=step,
            stability=stability,
            difficulty=difficulty,
            due=due,
            last_review=last_review,
        )

    def get_retrievability(self, current_datetime: datetime | None = None) -> float:
        """
        Calculates the Card object's current retrievability for a given date and time.

        The retrievability of a card is the predicted probability that the card is correctly recalled at the provided datetime.

        Args:
            current_datetime (datetime): The current date and time

        Returns:
            float: The retrievability of the Card object.
        """
        if self.last_review is None:
            return 0

        current_datetime = current_datetime or datetime.now(timezone.utc)
        elapsed_days = max(0, (current_datetime - self.last_review).days)
        return (1 + FACTOR * elapsed_days / self.stability) ** DECAY

    def SDR(
        self, current_datetime: datetime | None = None
    ) -> tuple[float, float, float]:
        assert self.stability is not None  # mypy
        assert self.difficulty is not None  # mypy
        return (
            self.stability,
            self.difficulty,
            self.get_retrievability(current_datetime),
        )


class ReviewLog:
    """
    Represents the log entry of a Card object that has been reviewed.

    Attributes:
        card (Card): Copy of the card object that was reviewed.
        rating (Rating): The rating given to the card during the review.
        review_datetime (datetime): The date and time of the review.
        review_duration (int | None): The number of miliseconds it took to review the card or None if unspecified.
    """

    card: Card
    rating: Rating
    review_datetime: datetime
    review_duration: int | None

    def __init__(
        self,
        card: Card,
        rating: Rating,
        review_datetime: datetime,
        review_duration: int | None = None,
    ) -> None:
        self.card = deepcopy(card)
        self.rating = rating
        self.review_datetime = review_datetime
        self.review_duration = review_duration

    def to_dict(self) -> dict[str, dict[str, Any] | int | str | None]:
        """
        Returns a JSON-serializable dictionary representation of the ReviewLog object.

        This method is specifically useful for storing ReviewLog objects in a database.

        Returns:
            dict: A dictionary representation of the Card object.
        """

        return_dict = {
            "card": self.card.to_dict(),
            "rating": self.rating.value,
            "review_datetime": self.review_datetime.isoformat(),
            "review_duration": self.review_duration,
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

        card = Card.from_dict(source_dict["card"])
        rating = Rating(int(source_dict["rating"]))
        review_datetime = datetime.fromisoformat(source_dict["review_datetime"])
        review_duration = source_dict["review_duration"]

        return ReviewLog(
            card=card,
            rating=rating,
            review_datetime=review_datetime,
            review_duration=review_duration,
        )


class Scheduler:
    """
    The FSRS scheduler.

    Enables the reviewing and future scheduling of cards according to the FSRS algorithm.

    Attributes:
        parameters (tuple[float, ...]): The 19 model weights of the FSRS scheduler.
        desired_retention (float): The desired retention rate of cards scheduled with the scheduler.
        learning_steps (tuple[timedelta, ...]): Small time intervals that schedule cards in the Learning state.
        relearning_steps (tuple[timedelta, ...]): Small time intervals that schedule cards in the Relearning state.
        maximum_interval (int): The maximum number of days a Review-state card can be scheduled into the future.
        enable_fuzzing (bool): Whether to apply a small amount of random 'fuzz' to calculated intervals.
    """

    parameters: tuple[float, ...]
    desired_retention: float
    learning_steps: tuple[timedelta, ...]
    relearning_steps: tuple[timedelta, ...]
    maximum_interval: int
    enable_fuzzing: bool

    def __init__(
        self,
        parameters: tuple[float, ...] | list[float] = (
            0.40255,
            1.18385,
            3.173,
            15.69105,
            7.1949,
            0.5345,
            1.4604,
            0.0046,
            1.54575,
            0.1192,
            1.01925,
            1.9395,
            0.11,
            0.29605,
            2.2698,
            0.2315,
            2.9898,
            0.51655,
            0.6621,
        ),
        desired_retention: float = 0.9,
        learning_steps: tuple[timedelta, ...] | list[timedelta] = (
            timedelta(minutes=1),
            timedelta(minutes=10),
        ),
        relearning_steps: tuple[timedelta, ...] | list[timedelta] = (
            timedelta(minutes=10),
        ),
        maximum_interval: int = 36500,
        enable_fuzzing: bool = True,
    ) -> None:
        self.parameters = tuple(parameters)
        self.w = self.parameters
        self.desired_retention = desired_retention
        self.learning_steps = tuple(learning_steps)
        self.relearning_steps = tuple(relearning_steps)
        self.maximum_interval = maximum_interval
        self.enable_fuzzing = enable_fuzzing

    def review_card(
        self,
        card: Card,
        rating: Rating,
        review_datetime: datetime | None = None,
        review_duration: int | None = None,
    ) -> tuple[Card, ReviewLog]:
        """
        Reviews a card with a given rating at a given time for a specified duration.

        Args:
            card (Card): The card being reviewed.
            rating (Rating): The chosen rating for the card being reviewed.
            review_datetime (datetime | None): The date and time of the review.
            review_duration (int | None): The number of miliseconds it took to review the card or None if unspecified.

        Returns:
            tuple[Card, ReviewLog]: A tuple containing the updated, reviewed card and its corresponding review log.

        Raises:
            ValueError: If the `review_datetime` argument is not timezone-aware and set to UTC.
        """

        if review_datetime is not None and (
            (review_datetime.tzinfo is None) or (review_datetime.tzinfo != timezone.utc)
        ):
            raise ValueError("datetime must be timezone-aware and set to UTC")

        if review_datetime is None:
            review_datetime = datetime.now(timezone.utc)

        card = deepcopy(card)

        review_log = ReviewLog(
            card=card,
            rating=rating,
            review_datetime=review_datetime,
            review_duration=review_duration,
        )

        card.stability = self._next_stability(card, rating, review_datetime)
        card.difficulty = self._next_difficulty(card.difficulty, rating)
        card.due = self._next_due(card, rating, review_datetime)
        card.last_review = review_datetime

        return card, review_log

    def _next_due(
        self, card: Card, rating: Rating, review_datetime: datetime
    ) -> datetime:
        next_interval = self._next_interval(card, rating)

        if self.enable_fuzzing and card.state == State.Review:
            next_interval = self._get_fuzzed_interval(next_interval)

        return review_datetime + next_interval

    def _next_interval(self, card: Card, rating: Rating) -> timedelta:
        ivl = (card.stability / FACTOR) * ((self.desired_retention ** (1 / DECAY)) - 1)
        next_interval = timedelta(days=clamp(round(ivl), 1, self.maximum_interval))

        def update_from_steps(steps: tuple[timedelta, ...]) -> timedelta:
            assert card.step is not None  # mypy

            if (
                card.step >= len(steps)
                or rating == Rating.Easy
                or (rating == Rating.Good and card.step + 1 == len(steps))
            ):
                card.state = State.Review
                card.step = None
                return next_interval
            elif rating == Rating.Hard:
                if card.step + 1 == len(steps):
                    return steps[card.step] * 1.5
                else:
                    return (steps[card.step] + steps[card.step + 1]) / 2.0
            elif rating == Rating.Again:
                card.step = 0
            else:  # Good with pending step
                card.step += 1

            return steps[card.step]

        if card.state == State.Learning:
            return update_from_steps(self.learning_steps)
        elif card.state == State.Relearning:
            return update_from_steps(self.relearning_steps)
        elif card.state == State.Review:
            if rating == Rating.Again and len(self.relearning_steps) > 0:
                card.state = State.Relearning
                card.step = 0
                return self.relearning_steps[card.step]
            else:
                return next_interval

    def to_dict(self) -> dict[str, Any]:
        """
        Returns a JSON-serializable dictionary representation of the Scheduler object.

        This method is specifically useful for storing Scheduler objects in a database.

        Returns:
            dict: A dictionary representation of the Scheduler object.
        """

        return_dict = {
            "parameters": self.parameters,
            "desired_retention": self.desired_retention,
            "learning_steps": [
                int(learning_step.total_seconds())
                for learning_step in self.learning_steps
            ],
            "relearning_steps": [
                int(relearning_step.total_seconds())
                for relearning_step in self.relearning_steps
            ],
            "maximum_interval": self.maximum_interval,
            "enable_fuzzing": self.enable_fuzzing,
        }

        return return_dict

    @staticmethod
    def from_dict(source_dict: dict[str, Any]) -> "Scheduler":
        """
        Creates a Scheduler object from an existing dictionary.

        Args:
            source_dict (dict[str, Any]): A dictionary representing an existing Scheduler object.

        Returns:
            Scheduler: A Scheduler object created from the provided dictionary.
        """

        parameters = source_dict["parameters"]
        desired_retention = source_dict["desired_retention"]
        learning_steps = [
            timedelta(seconds=learning_step)
            for learning_step in source_dict["learning_steps"]
        ]
        relearning_steps = [
            timedelta(seconds=relearning_step)
            for relearning_step in source_dict["relearning_steps"]
        ]
        maximum_interval = source_dict["maximum_interval"]
        enable_fuzzing = source_dict["enable_fuzzing"]

        return Scheduler(
            parameters=parameters,
            desired_retention=desired_retention,
            learning_steps=learning_steps,
            relearning_steps=relearning_steps,
            maximum_interval=maximum_interval,
            enable_fuzzing=enable_fuzzing,
        )

    def _next_stability(
        self, card: Card, rating: Rating, review_datetime: datetime
    ) -> float:
        if card.stability is None:
            return self._initial_stability(rating)

        if card.last_review and (review_datetime - card.last_review).days < 1:
            return self._short_term_stability(card.stability, rating)

        if rating == Rating.Again:
            return min(
                self._long_term_forget_stability(*card.SDR(review_datetime)),
                self._short_term_stability(card.stability, rating),
            )

        return self._recall_stability(*card.SDR(review_datetime), rating)

    # Methods using the model weights

    def _initial_stability(self, G: Rating) -> float:
        return [self.w[0], self.w[1], self.w[2], self.w[3]][G - 1]

    def _initial_difficulty(self, G: Rating) -> float:
        return self.w[4] - exp(self.w[5] * (G - 1)) + 1

    def _next_difficulty(self, D: float | None, G: Rating) -> float:
        if D is None:
            return self._initial_difficulty(G)

        D04 = self._initial_difficulty(Rating.Easy)
        delta_D = -self.w[6] * (G - 3)
        D_prime = D + delta_D * (10 - D) / 9
        D_double_prime = self.w[7] * D04 + (1 - self.w[7]) * D_prime

        return clamp(D_double_prime, 1, 10)

    def _recall_stability(self, S: float, D: float, R: float, rating: Rating) -> float:
        return S * (
            1
            + exp(self.w[8])
            * (11 - D)
            * pow(S, -self.w[9])
            * (exp((1 - R) * self.w[10]) - 1)
            * self._hard_penalty(rating)  # Is this correct,
            * self._easy_bonus(rating)  # or should it be outside the parentheses?
        )

    def _long_term_forget_stability(self, S: float, D: float, R: float) -> float:
        return (
            self.w[11]
            * pow(D, -self.w[12])
            * (pow(S + 1, self.w[13]) - 1)
            * exp((1 - R) * self.w[14])
        )

    def _hard_penalty(self, rating: Rating) -> float:
        return self.w[15] if rating == Rating.Hard else 1

    def _easy_bonus(self, rating: Rating) -> float:
        return self.w[16] if rating == Rating.Easy else 1

    def _short_term_stability(self, S: float, G: float) -> float:
        return S * exp(self.w[17] * (G - 3 + self.w[18]))

    def _get_fuzzed_interval(self, interval: timedelta) -> timedelta:
        """
        Takes the current calculated interval and adds a small amount of random fuzz to it.
        For example, a card that would've been due in 50 days, after fuzzing, might be due in 49, or 51 days.

        Args:
            interval (timedelta): The calculated next interval, before fuzzing.

        Returns:
            timedelta: The new interval, after fuzzing.
        """

        interval_days = interval.days

        if interval_days < 2.5:  # fuzz is not applied to intervals less than 2.5
            return interval

        def _get_fuzz_range(interval_days: int) -> tuple[int, int]:
            """
            Helper function that computes the possible upper and lower bounds of the interval after fuzzing.
            """

            delta = 1.0
            for fuzz_range in FUZZ_RANGES:
                delta += fuzz_range["factor"] * max(
                    min(interval_days, fuzz_range["end"]) - fuzz_range["start"], 0.0
                )

            min_ivl = int(round(interval_days - delta))
            max_ivl = int(round(interval_days + delta))

            # make sure the min_ivl and max_ivl fall into a valid range
            min_ivl = max(2, min_ivl)
            max_ivl = min(max_ivl, self.maximum_interval)
            min_ivl = min(min_ivl, max_ivl)

            return min_ivl, max_ivl

        min_ivl, max_ivl = _get_fuzz_range(interval_days)

        fuzzed_interval_days = (
            random.random() * (max_ivl - min_ivl + 1)
        ) + min_ivl  # the next interval is a random value between min_ivl and max_ivl

        fuzzed_interval_days = min(round(fuzzed_interval_days), self.maximum_interval)

        fuzzed_interval = timedelta(days=fuzzed_interval_days)

        return fuzzed_interval


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)
