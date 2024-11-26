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
from copy import deepcopy
from typing import Any
from enum import IntEnum

DECAY = -0.5
FACTOR = 0.9 ** (1 / DECAY) - 1

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
            Card: A Card object created from the provided dictionary.
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

    def __init__(self, card: Card, rating: Rating, review_datetime: datetime, review_duration: int | None = None) -> None:

        self.card = deepcopy(card)
        self.rating = rating
        self.review_datetime = review_datetime
        self.review_duration = review_duration

    def to_dict(self) -> dict[str, dict[str, Any] | int | str | None]:

        return_dict = {
            "card": self.card.to_dict(),
            "rating": self.rating.value,
            "review_datetime": self.review_datetime.isoformat(),
            "review_duration": self.review_duration,
        }

        return return_dict

    @staticmethod
    def from_dict(source_dict: dict[str, Any]) -> "ReviewLog":

        card = Card.from_dict(source_dict['card'])
        rating = Rating(int(source_dict['rating']))
        review_datetime = datetime.fromisoformat(source_dict['review_datetime'])
        review_duration = source_dict['review_duration']
    
        return ReviewLog(card=card, rating=rating, review_datetime=review_datetime, review_duration=review_duration)

class FSRSScheduler:
    """
    The FSRS scheduler.

    Enables the reviewing and future scheduling of cards according to the FSRS algorithm.

    Attributes:
        parameters (tuple[float, ...]): The 19 model weights of the FSRS scheduler.
        desired_retention (float): The desired retention rate of cards scheduled with the scheduler. Corresponds to the predicted probability of correctly recalling a card when it is next due.
        learning_steps (list[timedelta]): Small time intervals that schedule cards in the Learning state.
        relearning_steps (list[timedelta]): Small time intervals that schedule cards in the Relearning state.
        maximum_interval (int): The maximum number of days a Review-state card can be scheduled into the future.
    """

    parameters: tuple[float, ...]
    desired_retention: float
    learning_steps: list[timedelta]
    relearning_steps: list[timedelta]
    maximum_interval: int

    def __init__(self, 
                 parameters: tuple | list = (
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
                 ),
                 desired_retention: float = 0.9,
                 learning_steps: list[timedelta] = [timedelta(minutes=1), timedelta(minutes=10)],
                 relearning_steps: list[timedelta] = [timedelta(minutes=10)],
                 maximum_interval: int = 36500) -> None:

        self.parameters = tuple(parameters)
        self.desired_retention = desired_retention
        self.learning_steps = learning_steps
        self.relearning_steps = relearning_steps
        self.maximum_interval = maximum_interval

    def review_card(self, card: Card, rating: Rating, review_datetime: datetime | None = None, review_duration: int | None = None) -> tuple[Card, ReviewLog]:
        # TODO: implement review_card method
        pass

    def to_dict(self):
        pass

    @staticmethod
    def from_dict():
        pass