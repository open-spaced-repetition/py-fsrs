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
        card_id (int): The id of the card. Defaults to the epoch miliseconds of when the card was created.
        state (State): The card's current learning state.
        step (int | None): The card's current learning or relearning step or None if the card is in the Review state.
        stability (float | None): Core FSRS parameter used for scheduling.
        difficulty (float | None): Core FSRS parameter used for scheduling.
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
    
    def __init__(self,
                 card_id: int | None = None,
                 state: State = State.New,
                 step: int | None = None,
                 stability: float | None = None,
                 difficulty: float | None = None,
                 due: datetime | None = None,
                 last_review: datetime | None = None) -> None:
        
        if card_id is None:
            # epoch miliseconds of when the card was created
            card_id = int(datetime.now(timezone.utc).timestamp() * 1000)
        self.card_id = card_id

        self.state = state

        if self.state == State.New and step is None:
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
            "last_review": self.last_review.isoformat() if self.last_review else None
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

        card_id = int(source_dict['card_id'])
        state = State(int(source_dict['state']))
        step = source_dict['step']
        stability = float(source_dict['stability']) if source_dict['stability'] else None
        difficulty = float(source_dict['difficulty']) if source_dict['difficulty'] else None
        due = datetime.fromisoformat(source_dict['due'])
        last_review = datetime.fromisoformat(source_dict['last_review']) if source_dict['last_review'] else None

        return Card(card_id=card_id, state=state, step=step, stability=stability, difficulty=difficulty, due=due, last_review=last_review)

    def get_retrievability(self, current_datetime: datetime | None = None) -> float:
        """
        Calculates the Card object's current retrievability for a given date and time.

        The retrievability of a card is the predicted probability that the card is correctly recalled at the provided datetime.

        Args:
            current_datetime (datetime): The current date and time

        Returns:
            float: The retrievability of the Card object.
        """

        if current_datetime is None:
            current_datetime = datetime.now(timezone.utc)

        if self.state in (State.Learning, State.Review, State.Relearning):
            assert self.last_review is not None # mypy
            elapsed_days = max(0, (current_datetime - self.last_review).days)
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

    def to_dict(self) -> dict[str, Any]:

        return_dict = {
            "parameters": self.parameters,
            "desired_retention": self.desired_retention,
            "learning_steps": [int(learning_step.total_seconds()) for learning_step in self.learning_steps],
            "relearning_steps": [int(relearning_step.total_seconds()) for relearning_step in self.relearning_steps],
            "maximum_interval": self.maximum_interval
        }

        return return_dict

    @staticmethod
    def from_dict(source_dict: dict[str, Any]) -> "FSRSScheduler":

        parameters = source_dict['parameters']
        desired_retention = source_dict['desired_retention']
        learning_steps = [timedelta(seconds=learning_step) for learning_step in source_dict['learning_steps']]
        relearning_steps = [timedelta(seconds=relearning_step) for relearning_step in source_dict['relearning_steps']]
        maximum_interval = source_dict['maximum_interval']

        return FSRSScheduler(parameters=parameters, 
                             desired_retention=desired_retention,
                             learning_steps=learning_steps,
                             relearning_steps=relearning_steps,
                             maximum_interval=maximum_interval)
