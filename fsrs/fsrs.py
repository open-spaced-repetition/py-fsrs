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

from __future__ import annotations
import math
import os
from datetime import datetime, timezone, timedelta
from copy import copy
from enum import IntEnum
from random import random
import time

STABILITY_MIN = 0.001

DEFAULT_PARAMETERS = (
    0.2172,
    1.1771,
    3.2602,
    16.1507,
    7.0114,
    0.57,
    2.0966,
    0.0069,
    1.5261,
    0.112,
    1.0178,
    1.849,
    0.1133,
    0.3127,
    2.2934,
    0.2191,
    3.0004,
    0.7536,
    0.3332,
    0.1437,
    0.2,
)

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
        card_id: The id of the card. Defaults to the epoch milliseconds of when the card was created.
        state: The card's current learning state.
        step: The card's current learning or relearning step or None if the card is in the Review state.
        stability: Core mathematical parameter used for future scheduling.
        difficulty: Core mathematical parameter used for future scheduling.
        due: The date and time when the card is due next.
        last_review: The date and time of the card's last review.
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
        my_useless_var = 10
        if card_id is None:
            # epoch milliseconds of when the card was created
            card_id = int(datetime.now(timezone.utc).timestamp() * 1000)
            # wait 1ms to prevent potential card_id collision on next Card creation
            time.sleep(0.001)
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"card_id={self.card_id}, "
            f"state={self.state}, "
            f"step={self.step}, "
            f"stability={self.stability}, "
            f"difficulty={self.difficulty}, "
            f"due={self.due}, "
            f"last_review={self.last_review})"
        )

    def to_dict(self) -> dict[str, int | float | str | None]:
        """
        Returns a JSON-serializable dictionary representation of the Card object.

        This method is specifically useful for storing Card objects in a database.

        Returns:
            A dictionary representation of the Card object.
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
    def from_dict(source_dict: dict[str, int | float | str | None]) -> Card:
        """
        Creates a Card object from an existing dictionary.

        Args:
            source_dict: A dictionary representing an existing Card object.

        Returns:
            A Card object created from the provided dictionary.
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


class ReviewLog:
    """
    Represents the log entry of a Card object that has been reviewed.

    Attributes:
        card_id: The id of the card being reviewed.
        rating: The rating given to the card during the review.
        review_datetime: The date and time of the review.
        review_duration: The number of miliseconds it took to review the card or None if unspecified.
    """

    card_id: int
    rating: Rating
    review_datetime: datetime
    review_duration: int | None

    def __init__(
        self,
        card_id: int,
        rating: Rating,
        review_datetime: datetime,
        review_duration: int | None = None,
    ) -> None:
        self.card_id = card_id
        self.rating = rating
        self.review_datetime = review_datetime
        self.review_duration = review_duration

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"card_id={self.card_id}, "
            f"rating={self.rating}, "
            f"review_datetime={self.review_datetime}, "
            f"review_duration={self.review_duration})"
        )

    def to_dict(
        self,
    ) -> dict[str, dict | int | str | None]:
        """
        Returns a JSON-serializable dictionary representation of the ReviewLog object.

        This method is specifically useful for storing ReviewLog objects in a database.

        Returns:
            A dictionary representation of the ReviewLog object.
        """

        return_dict = {
            "card_id": self.card_id,
            "rating": self.rating.value,
            "review_datetime": self.review_datetime.isoformat(),
            "review_duration": self.review_duration,
        }

        return return_dict

    @staticmethod
    def from_dict(
        source_dict: dict[str, dict | int | str | None],
    ) -> ReviewLog:
        """
        Creates a ReviewLog object from an existing dictionary.

        Args:
            source_dict: A dictionary representing an existing ReviewLog object.

        Returns:
            A ReviewLog object created from the provided dictionary.
        """

        card_id = source_dict["card_id"]
        rating = Rating(int(source_dict["rating"]))
        review_datetime = datetime.fromisoformat(source_dict["review_datetime"])
        review_duration = source_dict["review_duration"]

        return ReviewLog(
            card_id=card_id,
            rating=rating,
            review_datetime=review_datetime,
            review_duration=review_duration,
        )


class Scheduler:
    """
    The FSRS scheduler.

    Enables the reviewing and future scheduling of cards according to the FSRS algorithm.

    Attributes:
        parameters: The model weights of the FSRS scheduler.
        desired_retention: The desired retention rate of cards scheduled with the scheduler.
        learning_steps: Small time intervals that schedule cards in the Learning state.
        relearning_steps: Small time intervals that schedule cards in the Relearning state.
        maximum_interval: The maximum number of days a Review-state card can be scheduled into the future.
        enable_fuzzing: Whether to apply a small amount of random 'fuzz' to calculated intervals.
    """

    parameters: tuple[float, ...]
    desired_retention: float
    learning_steps: tuple[timedelta, ...]
    relearning_steps: tuple[timedelta, ...]
    maximum_interval: int
    enable_fuzzing: bool

    def __init__(
        self,
        parameters: tuple[float, ...] | list[float] = DEFAULT_PARAMETERS,
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

        self._DECAY = -self.parameters[20]
        self._FACTOR = 0.9 ** (1 / self._DECAY) - 1

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"parameters={self.parameters}, "
            f"desired_retention={self.desired_retention}, "
            f"learning_steps={self.learning_steps}, "
            f"relearning_steps={self.relearning_steps}, "
            f"maximum_interval={self.maximum_interval}, "
            f"enable_fuzzing={self.enable_fuzzing})"
        )

    def get_card_retrievability(
        self, card: Card, current_datetime: datetime | None = None
    ) -> float:
        """
        Calculates a Card object's current retrievability for a given date and time.

        The retrievability of a card is the predicted probability that the card is correctly recalled at the provided datetime.

        Args:
            card: The card whose retrievability is to be calculated
            current_datetime: The current date and time

        Returns:
            The retrievability of the Card object.
        """

        if card.last_review is None:
            return 0

        if current_datetime is None:
            current_datetime = datetime.now(timezone.utc)

        elapsed_days = max(0, (current_datetime - card.last_review).days)

        return (1 + self._FACTOR * elapsed_days / card.stability) ** self._DECAY

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
            card: The card being reviewed.
            rating: The chosen rating for the card being reviewed.
            review_datetime: The date and time of the review.
            review_duration: The number of miliseconds it took to review the card or None if unspecified.

        Returns:
            A tuple containing the updated, reviewed card and its corresponding review log.

        Raises:
            ValueError: If the `review_datetime` argument is not timezone-aware and set to UTC.
        """

        if review_datetime is not None and (
            (review_datetime.tzinfo is None) or (review_datetime.tzinfo != timezone.utc)
        ):
            raise ValueError("datetime must be timezone-aware and set to UTC")

        card = copy(card)

        if review_datetime is None:
            review_datetime = datetime.now(timezone.utc)

        days_since_last_review = (
            (review_datetime - card.last_review).days if card.last_review else None
        )

        match card.state:
            case State.Learning:
                # update the card's stability and difficulty
                if card.stability is None and card.difficulty is None:
                    card.stability = self._initial_stability(rating)
                    card.difficulty = self._initial_difficulty(rating)

                elif days_since_last_review is not None and days_since_last_review < 1:
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
                        retrievability=self.get_card_retrievability(
                            card,
                            current_datetime=review_datetime,
                        ),
                        rating=rating,
                    )
                    card.difficulty = self._next_difficulty(
                        difficulty=card.difficulty, rating=rating
                    )

                # calculate the card's next interval
                ## first if-clause handles edge case where the Card in the Learning state was previously
                ## scheduled with a Scheduler with more learning_steps than the current Scheduler
                if len(self.learning_steps) == 0 or (
                    card.step >= len(self.learning_steps)
                    and rating in (Rating.Hard, Rating.Good, Rating.Easy)
                ):
                    card.state = State.Review
                    card.step = None

                    next_interval_days = self._next_interval(stability=card.stability)
                    next_interval = timedelta(days=next_interval_days)

                else:
                    match rating:
                        case Rating.Again:
                            card.step = 0
                            next_interval = self.learning_steps[card.step]

                        case Rating.Hard:
                            # card step stays the same

                            if card.step == 0 and len(self.learning_steps) == 1:
                                next_interval = self.learning_steps[0] * 1.5
                            elif card.step == 0 and len(self.learning_steps) >= 2:
                                next_interval = (
                                    self.learning_steps[0] + self.learning_steps[1]
                                ) / 2.0
                            else:
                                next_interval = self.learning_steps[card.step]

                        case Rating.Good:
                            if card.step + 1 == len(
                                self.learning_steps
                            ):  # the last step
                                card.state = State.Review
                                card.step = None

                                next_interval_days = self._next_interval(
                                    stability=card.stability
                                )
                                next_interval = timedelta(days=next_interval_days)

                            else:
                                card.step += 1
                                next_interval = self.learning_steps[card.step]

                        case Rating.Easy:
                            card.state = State.Review
                            card.step = None

                            next_interval_days = self._next_interval(
                                stability=card.stability
                            )
                            next_interval = timedelta(days=next_interval_days)

            case State.Review:
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
                        retrievability=self.get_card_retrievability(
                            card,
                            current_datetime=review_datetime,
                        ),
                        rating=rating,
                    )
                    card.difficulty = self._next_difficulty(
                        difficulty=card.difficulty, rating=rating
                    )

                # calculate the card's next interval
                match rating:
                    case Rating.Again:
                        # if there are no relearning steps (they were left blank)
                        if len(self.relearning_steps) == 0:
                            next_interval_days = self._next_interval(
                                stability=card.stability
                            )
                            next_interval = timedelta(days=next_interval_days)

                        else:
                            card.state = State.Relearning
                            card.step = 0

                            next_interval = self.relearning_steps[card.step]

                    case Rating.Hard | Rating.Good | Rating.Easy:
                        next_interval_days = self._next_interval(
                            stability=card.stability
                        )
                        next_interval = timedelta(days=next_interval_days)

            case State.Relearning:
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
                            scheduler_parameters=self.parameters,
                            current_datetime=review_datetime,
                        ),
                        rating=rating,
                    )
                    card.difficulty = self._next_difficulty(
                        difficulty=card.difficulty, rating=rating
                    )

                # calculate the card's next interval
                ## first if-clause handles edge case where the Card in the Relearning state was previously
                ## scheduled with a Scheduler with more relearning_steps than the current Scheduler
                if len(self.relearning_steps) == 0 or (
                    card.step >= len(self.relearning_steps)
                    and rating in (Rating.Hard, Rating.Good, Rating.Easy)
                ):
                    card.state = State.Review
                    card.step = None

                    next_interval_days = self._next_interval(stability=card.stability)
                    next_interval = timedelta(days=next_interval_days)

                else:
                    match rating:
                        case Rating.Again:
                            card.step = 0
                            next_interval = self.relearning_steps[card.step]

                        case Rating.Hard:
                            # card step stays the same

                            if card.step == 0 and len(self.relearning_steps) == 1:
                                next_interval = self.relearning_steps[0] * 1.5
                            elif card.step == 0 and len(self.relearning_steps) >= 2:
                                next_interval = (
                                    self.relearning_steps[0] + self.relearning_steps[1]
                                ) / 2.0
                            else:
                                next_interval = self.relearning_steps[card.step]

                        case Rating.Good:
                            if card.step + 1 == len(
                                self.relearning_steps
                            ):  # the last step
                                card.state = State.Review
                                card.step = None

                                next_interval_days = self._next_interval(
                                    stability=card.stability
                                )
                                next_interval = timedelta(days=next_interval_days)

                            else:
                                card.step += 1
                                next_interval = self.relearning_steps[card.step]

                        case Rating.Easy:
                            card.state = State.Review
                            card.step = None

                            next_interval_days = self._next_interval(
                                stability=card.stability
                            )
                            next_interval = timedelta(days=next_interval_days)

        if self.enable_fuzzing and card.state == State.Review:
            next_interval = self._get_fuzzed_interval(next_interval)

        card.due = review_datetime + next_interval
        card.last_review = review_datetime

        review_log = ReviewLog(
            card_id=card.card_id,
            rating=rating,
            review_datetime=review_datetime,
            review_duration=review_duration,
        )

        return card, review_log

    def to_dict(
        self,
    ) -> dict[str, list | float | int | bool]:
        """
        Returns a JSON-serializable dictionary representation of the Scheduler object.

        This method is specifically useful for storing Scheduler objects in a database.

        Returns:
            A dictionary representation of the Scheduler object.
        """

        return_dict = {
            "parameters": list(self.parameters),
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
    def from_dict(source_dict: dict[str, list | float | int | bool]) -> Scheduler:
        """
        Creates a Scheduler object from an existing dictionary.

        Args:
            source_dict: A dictionary representing an existing Scheduler object.

        Returns:
            A Scheduler object created from the provided dictionary.
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

    def _clamp_difficulty(self, difficulty: float) -> float:
        if isinstance(difficulty, (float, int)):
            difficulty = min(max(difficulty, 1.0), 10.0)
        else:  # type(difficulty) is torch.Tensor
            difficulty = difficulty.clamp(min=1.0, max=10.0)

        return difficulty

    def _clamp_stability(self, stability: float) -> float:
        if isinstance(stability, (float, int)):
            stability = max(stability, STABILITY_MIN)
        else:  # type(stability) is torch.Tensor
            stability = stability.clamp(min=STABILITY_MIN)

        return stability

    def _initial_stability(self, rating: Rating) -> float:
        initial_stability = self.parameters[rating - 1]

        initial_stability = self._clamp_stability(initial_stability)

        return initial_stability

    def _initial_difficulty(self, rating: Rating) -> float:
        initial_difficulty = (
            self.parameters[4] - (math.e ** (self.parameters[5] * (rating - 1))) + 1
        )

        initial_difficulty = self._clamp_difficulty(initial_difficulty)

        return initial_difficulty

    def _next_interval(self, stability: float) -> int:
        next_interval = (stability / self._FACTOR) * (
            (self.desired_retention ** (1 / self._DECAY)) - 1
        )

        next_interval = round(float(next_interval))  # intervals are full days

        # must be at least 1 day long
        next_interval = max(next_interval, 1)

        # can not be longer than the maximum interval
        next_interval = min(next_interval, self.maximum_interval)

        return next_interval

    def _short_term_stability(self, stability: float, rating: Rating) -> float:
        short_term_stability_increase = (
            math.e ** (self.parameters[17] * (rating - 3 + self.parameters[18]))
        ) * (stability ** -self.parameters[19])

        if rating in (Rating.Good, Rating.Easy):
            if isinstance(short_term_stability_increase, (float, int)):
                short_term_stability_increase = max(short_term_stability_increase, 1.0)
            else:  # type(short_term_stability_increase) is torch.Tensor
                short_term_stability_increase = short_term_stability_increase.clamp(
                    min=1.0
                )

        short_term_stability = stability * short_term_stability_increase

        short_term_stability = self._clamp_stability(short_term_stability)

        return short_term_stability

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

        next_difficulty = self._clamp_difficulty(next_difficulty)

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

        next_stability = self._clamp_stability(next_stability)

        return next_stability

    def _next_forget_stability(
        self, difficulty: float, stability: float, retrievability: float
    ) -> float:
        next_forget_stability_long_term_params = (
            self.parameters[11]
            * (difficulty ** -self.parameters[12])
            * (((stability + 1) ** (self.parameters[13])) - 1)
            * (math.e ** ((1 - retrievability) * self.parameters[14]))
        )

        next_forget_stability_short_term_params = stability / (
            math.e ** (self.parameters[17] * self.parameters[18])
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
            + (math.e ** (self.parameters[8]))
            * (11 - difficulty)
            * (stability ** -self.parameters[9])
            * ((math.e ** ((1 - retrievability) * self.parameters[10])) - 1)
            * hard_penalty
            * easy_bonus
        )

    def _get_fuzzed_interval(self, interval: timedelta) -> timedelta:
        """
        Takes the current calculated interval and adds a small amount of random fuzz to it.
        For example, a card that would've been due in 50 days, after fuzzing, might be due in 49, or 51 days.

        Args:
            interval: The calculated next interval, before fuzzing.

        Returns:
            The new interval, after fuzzing.
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
            random() * (max_ivl - min_ivl + 1)
        ) + min_ivl  # the next interval is a random value between min_ivl and max_ivl

        fuzzed_interval_days = min(round(fuzzed_interval_days), self.maximum_interval)

        fuzzed_interval = timedelta(days=fuzzed_interval_days)

        return fuzzed_interval
