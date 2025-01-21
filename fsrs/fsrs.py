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
from datetime import datetime, timezone, timedelta
from copy import copy, deepcopy
from enum import IntEnum
from random import random, Random

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DEFAULT_PARAMETERS = [
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
]

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
        card_id (int): The id of the card. Defaults to the epoch microseconds of when the card was created.
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
            # epoch microseconds of when the card was created
            card_id = int(datetime.now(timezone.utc).timestamp() * 1000000)
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
    def from_dict(source_dict: dict[str, int | float | str | None]) -> Card:
        """
        Creates a Card object from an existing dictionary.

        Args:
            source_dict (dict[str, int | float | str | None]): A dictionary representing an existing Card object.

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
        self.card = copy(card)
        self.rating = rating
        self.review_datetime = review_datetime
        self.review_duration = review_duration

    def to_dict(
        self,
    ) -> dict[str, dict | int | str | None]:
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
    def from_dict(
        source_dict: dict[str, dict | int | str | None],
    ) -> ReviewLog:
        """
        Creates a ReviewLog object from an existing dictionary.

        Args:
            source_dict (dict[str, dict | int | str | None]): A dictionary representing an existing ReviewLog object.

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

        card = copy(card)

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

    def to_dict(
        self,
    ) -> dict[str, list | float | int | bool]:
        """
        Returns a JSON-serializable dictionary representation of the Scheduler object.

        This method is specifically useful for storing Scheduler objects in a database.

        Returns:
            dict: A dictionary representation of the Scheduler object.
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
            source_dict (dict[str, list | float | int | bool]): A dictionary representing an existing Scheduler object.

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
            self.parameters[4] - (math.e ** (self.parameters[5] * (rating - 1))) + 1
        )

        # bound initial_difficulty between 1 and 10
        initial_difficulty = min(max(initial_difficulty, 1.0), 10.0)

        return initial_difficulty

    def _next_interval(self, stability: float) -> int:
        next_interval = (stability / FACTOR) * (
            (self.desired_retention ** (1 / DECAY)) - 1
        )

        next_interval = round(float(next_interval))  # intervals are full days

        # must be at least 1 day long
        next_interval = max(next_interval, 1)

        # can not be longer than the maximum interval
        next_interval = min(next_interval, self.maximum_interval)

        return next_interval

    def _short_term_stability(self, stability: float, rating: Rating) -> float:
        return stability * (
            math.e ** (self.parameters[17] * (rating - 3 + self.parameters[18]))
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
            random() * (max_ivl - min_ivl + 1)
        ) + min_ivl  # the next interval is a random value between min_ivl and max_ivl

        fuzzed_interval_days = min(round(fuzzed_interval_days), self.maximum_interval)

        fuzzed_interval = timedelta(days=fuzzed_interval_days)

        return fuzzed_interval


if TORCH_AVAILABLE:
    from torch.nn import BCELoss
    from torch import optim

    # weight clipping
    S_MIN = 0.01
    INIT_S_MAX = 100.0
    lower_bounds = torch.tensor(
        [
            S_MIN,
            S_MIN,
            S_MIN,
            S_MIN,
            1.0,
            0.1,
            0.1,
            0.0,
            0.0,
            0.0,
            0.01,
            0.1,
            0.01,
            0.01,
            0.01,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        dtype=torch.float64,
    )
    upper_bounds = torch.tensor(
        [
            INIT_S_MAX,
            INIT_S_MAX,
            INIT_S_MAX,
            INIT_S_MAX,
            10.0,
            4.0,
            4.0,
            0.75,
            4.5,
            0.8,
            3.5,
            5.0,
            0.25,
            0.9,
            4.0,
            1.0,
            6.0,
            2.0,
            2.0,
        ],
        dtype=torch.float64,
    )

    # hyper parameters
    num_epochs = 5
    mini_batch_size = 512
    learning_rate = 4e-2
    max_seq_len = (
        64  # up to the first 64 reviews of each card are used for optimization
    )

    class Optimizer:
        """
        The FSRS optimizer.

        Enables the optimization of FSRS scheduler parameters from existing review logs for more accurate interval calculations.

        Attributes:
            review_logs (tuple[ReviewLog, ...]): A collection of previous ReviewLog objects from a user.
            _revlogs_train (dict): The collection of review logs, sorted and formatted for optimization.
        """

        review_logs: tuple[ReviewLog, ...]
        _revlogs_train: dict

        def __init__(
            self, review_logs: tuple[ReviewLog, ...] | list[ReviewLog]
        ) -> None:
            """
            Initializes the Optimizer with a set of ReviewLogs. Also formats an copy of the review logs for optimization.

            Note that the ReviewLogs provided by the user don't need to be in order.
            """

            def _format_revlogs() -> dict:
                """
                Sorts and converts the tuple of ReviewLog objects to a dictionary format for optimizing
                """

                revlogs_train = {}
                for review_log in self.review_logs:
                    # pull data out of current ReviewLog object
                    card_id = review_log.card.card_id
                    rating = review_log.rating
                    review_datetime = review_log.review_datetime
                    review_duration = review_log.review_duration

                    # if the card was rated Again, it was not recalled
                    recall = 0 if rating == Rating.Again else 1

                    # as a ML problem, [x, y] = [ [review_datetime, rating, review_duration], recall ]
                    datum = [[review_datetime, rating, review_duration], recall]

                    if card_id not in revlogs_train:
                        revlogs_train[card_id] = []

                    revlogs_train[card_id].append((datum))
                    revlogs_train[card_id] = sorted(
                        revlogs_train[card_id], key=lambda x: x[0][0]
                    )  # keep reviews sorted

                # convert the timestamps in the json from isoformat to datetime variables
                for key, values in revlogs_train.items():
                    for entry in values:
                        entry[0][0] = datetime.fromisoformat(entry[0][0])

                # sort the dictionary in order of when each card history starts
                revlogs_train = dict(sorted(revlogs_train.items()))

                return revlogs_train

            self.review_logs = deepcopy(tuple(review_logs))

            # format the ReviewLog data for optimization
            self._revlogs_train = _format_revlogs()

        def _compute_batch_loss(self, parameters: list[float]) -> float:
            """
            Computes the current total loss for the entire batch of review logs.
            """

            card_ids = list(self._revlogs_train.keys())
            params = torch.tensor(parameters, dtype=torch.float64)
            loss_fn = BCELoss()
            scheduler = Scheduler(parameters=params)
            step_losses = []

            for card_id in card_ids:
                card_review_history = self._revlogs_train[card_id][:max_seq_len]

                for i in range(len(card_review_history)):
                    review = card_review_history[i]

                    x_date = review[0][0]
                    y_retrievability = review[1]
                    u_rating = review[0][1]

                    if i == 0:
                        card = Card(due=x_date)

                    y_pred_retrievability = card.get_retrievability(x_date)
                    y_retrievability = torch.tensor(
                        y_retrievability, dtype=torch.float64
                    )

                    if card.state == State.Review:
                        step_loss = loss_fn(y_pred_retrievability, y_retrievability)
                        step_losses.append(step_loss)

                    card, _ = scheduler.review_card(
                        card=card,
                        rating=u_rating,
                        review_datetime=x_date,
                        review_duration=None,
                    )

            batch_loss = torch.sum(torch.stack(step_losses))
            batch_loss = batch_loss.item() / len(step_losses)

            return batch_loss

        def compute_optimal_parameters(self) -> list[float]:
            """
            Computes a set of 19 optimized parameters for the FSRS scheduler and returns it as a list of floats.

            High level explanation of optimization:
            ---------------------------------------
            FSRS is a many-to-many sequence model where the "State" at each step is a Card object at a given point in time,
            the input is the time of the review and the output is the predicted retrievability of the card at the time of review.

            Each card's review history can be thought of as a sequence, each review as a step and each collection of card review histories
            as a batch.

            The loss is computed by comparing the predicted retrievability of the Card at each step with whether the Card was actually
            sucessfully recalled or not (0/1).

            Finally, the card objects at each step in their sequences are updated using the 19 current parameters of the Scheduler
            as well as the rating given to that card by the user. The 19 parameters of the Scheduler is what is being optimized.
            """

            def _num_reviews() -> int:
                """
                Computes how many Review-state reviews there are in the dataset.
                Only the loss from Review-state reviews count for optimization and their number must
                be computed in advance to properly initialize the Cosine Annealing learning rate scheduler.
                """

                scheduler = Scheduler()
                num_reviews = 0
                # iterate through the card review histories
                card_ids = list(self._revlogs_train.keys())
                for card_id in card_ids:
                    card_review_history = self._revlogs_train[card_id][:max_seq_len]

                    # iterate through the current Card's review history
                    for i in range(len(card_review_history)):
                        review = card_review_history[i]

                        review_datetime = review[0][0]
                        rating = review[0][1]

                        # if this is the first review, create the Card object
                        if i == 0:
                            card = Card(due=review_datetime)

                        # only Review-state reviews count
                        if card.state == State.Review:
                            num_reviews += 1

                        card, _ = scheduler.review_card(
                            card=card,
                            rating=rating,
                            review_datetime=review_datetime,
                            review_duration=None,
                        )

                return num_reviews

            def _update_parameters(
                step_losses: list,
                adam_optimizer: torch.optim.Adam,
                params: torch.Tensor,
                lr_scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
            ) -> None:
                """
                Computes and updates the current FSRS parameters based on the step losses. Also updates the learning rate scheduler.
                """

                # Backpropagate through the loss
                mini_batch_loss = torch.sum(torch.stack(step_losses))
                adam_optimizer.zero_grad()  # clear previous gradients
                mini_batch_loss.backward()  # compute gradients
                adam_optimizer.step()  # Update parameters

                # clamp the weights in place without modifying the computational graph
                with torch.no_grad():
                    params.clamp_(min=lower_bounds, max=upper_bounds)

                # update the learning rate
                lr_scheduler.step()

            # set local random seed for reproducibility
            rng = Random(42)

            card_ids = list(self._revlogs_train.keys())

            num_reviews = _num_reviews()

            if num_reviews < mini_batch_size:
                return DEFAULT_PARAMETERS

            # Define FSRS Scheduler parameters as torch tensors with gradients
            params = torch.tensor(
                DEFAULT_PARAMETERS, requires_grad=True, dtype=torch.float64
            )

            loss_fn = BCELoss()
            adam_optimizer = optim.Adam([params], lr=learning_rate)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=adam_optimizer,
                T_max=math.ceil(num_reviews / mini_batch_size) * num_epochs,
            )

            best_params = None
            best_loss = math.inf
            # iterate through the epochs
            for j in range(num_epochs):
                # randomly shuffle the order of which Card's review histories get computed first
                # at the beginning of each new epoch
                rng.shuffle(card_ids)

                # initialize new scheduler with updated parameters each epoch
                scheduler = Scheduler(parameters=params)

                # stores the computed loss of each individual review
                step_losses = []

                # iterate through the card review histories (sequences)
                for card_id in card_ids:
                    card_review_history = self._revlogs_train[card_id][:max_seq_len]

                    # iterate through the current Card's review history (steps)
                    for i in range(len(card_review_history)):
                        review = card_review_history[i]

                        # input
                        x_date = review[0][0]
                        # target
                        y_retrievability = review[1]
                        # update
                        u_rating = review[0][1]

                        # if this is the first review, create the Card object
                        if i == 0:
                            card = Card(due=x_date)

                        # predicted target
                        y_pred_retrievability = card.get_retrievability(x_date)
                        y_retrievability = torch.tensor(
                            y_retrievability, dtype=torch.float64
                        )

                        # only compute step-loss on Review-state cards
                        if card.state == State.Review:
                            step_loss = loss_fn(y_pred_retrievability, y_retrievability)
                            step_losses.append(step_loss)

                        # update the card's state
                        card, _ = scheduler.review_card(
                            card=card,
                            rating=u_rating,
                            review_datetime=x_date,
                            review_duration=None,
                        )

                        # take a gradient step after each mini-batch
                        if len(step_losses) == mini_batch_size:
                            _update_parameters(
                                step_losses=step_losses,
                                adam_optimizer=adam_optimizer,
                                params=params,
                                lr_scheduler=lr_scheduler,
                            )

                            # update the scheduler's with the new parameters
                            scheduler = Scheduler(parameters=params)
                            # clear the step losses for next batch
                            step_losses = []

                            # remove gradient history from tensor card parameters for next batch
                            card.stability = card.stability.detach()
                            card.difficulty = card.difficulty.detach()

                # update params on remaining review logs
                if len(step_losses) > 0:
                    _update_parameters(
                        step_losses=step_losses,
                        adam_optimizer=adam_optimizer,
                        params=params,
                        lr_scheduler=lr_scheduler,
                    )

                # compute the current batch loss after each epoch
                detached_params = [
                    x.detach().item() for x in list(params.detach())
                ]  # convert to floats
                with torch.no_grad():
                    epoch_batch_loss = self._compute_batch_loss(
                        parameters=detached_params
                    )

                # if the batch loss is better with the current parameters, update the current best parameters
                if epoch_batch_loss < best_loss:
                    best_loss = epoch_batch_loss
                    best_params = detached_params

            return best_params

else:

    class Optimizer:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("The Optimizer class requires torch be installed.")
