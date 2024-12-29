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

import math
from datetime import datetime, timezone, timedelta
from copy import deepcopy
from typing import Any
from enum import IntEnum
import random

DECAY = -0.5
FACTOR = 0.9 ** (1 / DECAY) - 1

FUZZ_RANGES = [
    {
        "start": 2.5,
        "end": 7.0,
        "factor": 0.15,
    },
    {
        "start": 7.0,
        "end": 20.0,
        "factor": 0.1,
    },
    {
        "start": 20.0,
        "end": math.inf,
        "factor": 0.05,
    },
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

        if current_datetime is None:
            current_datetime = datetime.now(timezone.utc)

        elapsed_days = max(0, (current_datetime - self.last_review).days)

        return (1 + FACTOR * elapsed_days / self.stability) ** DECAY


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

        card = deepcopy(card)

        if review_datetime is None:
            review_datetime = datetime.now(timezone.utc)

        days_since_last_review = (
            (review_datetime - card.last_review).days if card.last_review else None
        )

        review_log = ReviewLog(
            card=card,
            rating=rating,
            review_datetime=review_datetime,
            review_duration=review_duration,
        )

        if card.state == State.Learning:
            assert type(card.step) is int

            # update the card's stability and difficulty
            if card.stability is None and card.difficulty is None:
                card.stability = self._initial_stability(rating)
                card.difficulty = self._initial_difficulty(rating)

            elif days_since_last_review is not None and days_since_last_review < 1:
                assert type(card.stability) is float  # mypy
                assert type(card.difficulty) is float  # mypy
                card.stability = self._short_term_stability(
                    stability=card.stability, rating=rating
                )
                card.difficulty = self._next_difficulty(
                    difficulty=card.difficulty, rating=rating
                )

            else:
                assert type(card.stability) is float  # mypy
                assert type(card.difficulty) is float  # mypy
                card.stability = self._next_stability(
                    difficulty=card.difficulty,
                    stability=card.stability,
                    retrievability=card.get_retrievability(
                        current_datetime=review_datetime
                    ),
                    rating=rating,
                )
                card.difficulty = self._next_difficulty(
                    difficulty=card.difficulty, rating=rating
                )

            # calculate the card's next interval
            # len(self.learning_steps) == 0: no learning steps defined so move card to Review state
            # card.step > len(self.learning_steps): handles the edge-case when a card was originally scheduled with a scheduler with more
            # learning steps than the current scheduler
            if len(self.learning_steps) == 0 or card.step > len(self.learning_steps):
                card.state = State.Review
                card.step = None

                next_interval_days = self._next_interval(stability=card.stability)
                next_interval = timedelta(days=next_interval_days)

            else:
                if rating == Rating.Again:
                    card.step = 0
                    next_interval = self.learning_steps[card.step]

                elif rating == Rating.Hard:
                    # card step stays the same

                    if card.step == 0 and len(self.learning_steps) == 1:
                        next_interval = self.learning_steps[0] * 1.5
                    elif card.step == 0 and len(self.learning_steps) >= 2:
                        next_interval = (
                            self.learning_steps[0] + self.learning_steps[1]
                        ) / 2.0
                    else:
                        next_interval = self.learning_steps[card.step]

                elif rating == Rating.Good:
                    if card.step + 1 == len(self.learning_steps):  # the last step
                        card.state = State.Review
                        card.step = None

                        next_interval_days = self._next_interval(
                            stability=card.stability
                        )
                        next_interval = timedelta(days=next_interval_days)

                    else:
                        card.step += 1
                        next_interval = self.learning_steps[card.step]

                elif rating == Rating.Easy:
                    card.state = State.Review
                    card.step = None

                    next_interval_days = self._next_interval(stability=card.stability)
                    next_interval = timedelta(days=next_interval_days)

        elif card.state == State.Review:
            assert type(card.stability) is float  # mypy
            assert type(card.difficulty) is float  # mypy

            # update the card's stability and difficulty
            if days_since_last_review is not None and days_since_last_review < 1:
                card.stability = self._short_term_stability(
                    stability=card.stability, rating=rating
                )
                card.difficulty = self._next_difficulty(
                    difficulty=card.difficulty, rating=rating
                )

            else:
                card.stability = self._next_stability(
                    difficulty=card.difficulty,
                    stability=card.stability,
                    retrievability=card.get_retrievability(
                        current_datetime=review_datetime
                    ),
                    rating=rating,
                )
                card.difficulty = self._next_difficulty(
                    difficulty=card.difficulty, rating=rating
                )

            # calculate the card's next interval
            if rating == Rating.Again:
                # if there are no relearning steps (they were left blank)
                if len(self.relearning_steps) == 0:
                    next_interval_days = self._next_interval(stability=card.stability)
                    next_interval = timedelta(days=next_interval_days)

                else:
                    card.state = State.Relearning
                    card.step = 0

                    next_interval = self.relearning_steps[card.step]

            elif rating in (Rating.Hard, Rating.Good, Rating.Easy):
                next_interval_days = self._next_interval(stability=card.stability)
                next_interval = timedelta(days=next_interval_days)

        elif card.state == State.Relearning:
            assert type(card.step) is int
            assert type(card.stability) is float  # mypy
            assert type(card.difficulty) is float  # mypy

            # update the card's stability and difficulty
            if days_since_last_review is not None and days_since_last_review < 1:
                card.stability = self._short_term_stability(
                    stability=card.stability, rating=rating
                )
                card.difficulty = self._next_difficulty(
                    difficulty=card.difficulty, rating=rating
                )

            else:
                card.stability = self._next_stability(
                    difficulty=card.difficulty,
                    stability=card.stability,
                    retrievability=card.get_retrievability(
                        current_datetime=review_datetime
                    ),
                    rating=rating,
                )
                card.difficulty = self._next_difficulty(
                    difficulty=card.difficulty, rating=rating
                )

            # calculate the card's next interval
            # len(self.relearning_steps) == 0: no relearning steps defined so move card to Review state
            # card.step > len(self.relearning_steps): handles the edge-case when a card was originally scheduled with a scheduler with more
            # relearning steps than the current scheduler
            if len(self.relearning_steps) == 0 or card.step > len(
                self.relearning_steps
            ):
                card.state = State.Review
                card.step = None

                next_interval_days = self._next_interval(stability=card.stability)
                next_interval = timedelta(days=next_interval_days)

            else:
                if rating == Rating.Again:
                    card.step = 0
                    next_interval = self.relearning_steps[card.step]

                elif rating == Rating.Hard:
                    # card step stays the same

                    if card.step == 0 and len(self.relearning_steps) == 1:
                        next_interval = self.relearning_steps[0] * 1.5
                    elif card.step == 0 and len(self.relearning_steps) >= 2:
                        next_interval = (
                            self.relearning_steps[0] + self.relearning_steps[1]
                        ) / 2.0
                    else:
                        next_interval = self.relearning_steps[card.step]

                elif rating == Rating.Good:
                    if card.step + 1 == len(self.relearning_steps):  # the last step
                        card.state = State.Review
                        card.step = None

                        next_interval_days = self._next_interval(
                            stability=card.stability
                        )
                        next_interval = timedelta(days=next_interval_days)

                    else:
                        card.step += 1
                        next_interval = self.relearning_steps[card.step]

                elif rating == Rating.Easy:
                    card.state = State.Review
                    card.step = None

                    next_interval_days = self._next_interval(stability=card.stability)
                    next_interval = timedelta(days=next_interval_days)

        if self.enable_fuzzing and card.state == State.Review:
            next_interval = self._get_fuzzed_interval(next_interval)

        card.due = review_datetime + next_interval
        card.last_review = review_datetime

        return card, review_log

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

    def _initial_stability(self, rating: Rating) -> float:
        initial_stability = self.parameters[rating - 1]

        # initial_stability >= 0.1
        initial_stability = max(initial_stability, 0.1)

        return initial_stability

    def _initial_difficulty(self, rating: Rating) -> float:
        initial_difficulty = (
            self.parameters[4] - math.exp(self.parameters[5] * (rating - 1)) + 1
        )

        # bound initial_difficulty between 1 and 10
        initial_difficulty = min(max(initial_difficulty, 1.0), 10.0)

        return initial_difficulty

    def _next_interval(self, stability: float) -> int:
        next_interval = (stability / FACTOR) * (
            (self.desired_retention ** (1 / DECAY)) - 1
        )

        next_interval = round(next_interval)  # intervals are full days

        # must be at least 1 day long
        next_interval = max(next_interval, 1)

        # can not be longer than the maximum interval
        next_interval = min(next_interval, self.maximum_interval)

        return next_interval

    def _short_term_stability(self, stability: float, rating: Rating) -> float:
        return stability * math.exp(
            self.parameters[17] * (rating - 3 + self.parameters[18])
        )

    def _next_difficulty(self, difficulty: float, rating: Rating) -> float:
        def _linear_damping(delta_difficulty: float, difficulty: float) -> float:
            return (10.0 - difficulty) * delta_difficulty / 9.0

        def _mean_reversion(arg_1: float, arg_2: float) -> float:
            return self.parameters[7] * arg_1 + (1 - self.parameters[7]) * arg_2

        arg_1 = self._initial_difficulty(Rating.Easy)

        delta_difficulty = -(self.parameters[6] * (rating - 3))
        arg_2 = difficulty + _linear_damping(
            delta_difficulty=delta_difficulty, difficulty=difficulty
        )

        next_difficulty = _mean_reversion(arg_1=arg_1, arg_2=arg_2)

        # bound next_difficulty between 1 and 10
        next_difficulty = min(max(next_difficulty, 1.0), 10.0)

        return next_difficulty

    def _next_stability(
        self, difficulty: float, stability: float, retrievability: float, rating: Rating
    ) -> float:
        if rating == Rating.Again:
            next_stability = self._next_forget_stability(
                difficulty=difficulty,
                stability=stability,
                retrievability=retrievability,
            )

        elif rating in (Rating.Hard, Rating.Good, Rating.Easy):
            next_stability = self._next_recall_stability(
                difficulty=difficulty,
                stability=stability,
                retrievability=retrievability,
                rating=rating,
            )

        return next_stability

    def _next_forget_stability(
        self, difficulty: float, stability: float, retrievability: float
    ) -> float:
        next_forget_stability_long_term_params = (
            self.parameters[11]
            * math.pow(difficulty, -self.parameters[12])
            * (math.pow(stability + 1, self.parameters[13]) - 1)
            * math.exp((1 - retrievability) * self.parameters[14])
        )

        next_forget_stability_short_term_params = stability / math.exp(
            self.parameters[17] * self.parameters[18]
        )

        return min(
            next_forget_stability_long_term_params,
            next_forget_stability_short_term_params,
        )

    def _next_recall_stability(
        self, difficulty: float, stability: float, retrievability: float, rating: Rating
    ) -> float:
        hard_penalty = self.parameters[15] if rating == Rating.Hard else 1
        easy_bonus = self.parameters[16] if rating == Rating.Easy else 1

        return stability * (
            1
            + math.exp(self.parameters[8])
            * (11 - difficulty)
            * math.pow(stability, -self.parameters[9])
            * (math.exp((1 - retrievability) * self.parameters[10]) - 1)
            * hard_penalty
            * easy_bonus
        )

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
