from copy import deepcopy
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any


class Rating(IntEnum):
    """Rating given when reviewing a card."""

    Again = 1
    Hard = 2
    Good = 3
    Easy = 4


class State(IntEnum):
    """Learning state of a card."""

    Learning = 1
    Review = 2
    Relearning = 3


class Card:
    """
    A flashcard in the FSRS system.

    Attributes:
        card_id: ID of the card. Defaults to current epoch milliseconds.
        state: Current learning state.
        step: Current learning/relearning step (None if in Review state).
        stability: Core parameter for scheduling.
        difficulty: Core parameter for scheduling.
        due: When the card is due next.
        last_review: When the card was last reviewed.
    """

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
        self.card_id = card_id or int(datetime.now(timezone.utc).timestamp() * 1000)
        self.state = state
        self.step = 0 if state == State.Learning and step is None else step
        self.stability = stability
        self.difficulty = difficulty
        self.due = due or datetime.now(timezone.utc)
        self.last_review = last_review

    def to_dict(self) -> dict[str, int | float | str | None]:
        """Convert card to a JSON-serializable dictionary."""
        return {
            "card_id": self.card_id,
            "state": self.state.value,
            "step": self.step,
            "stability": self.stability,
            "difficulty": self.difficulty,
            "due": self.due.isoformat(),
            "last_review": self.last_review.isoformat() if self.last_review else None,
        }

    @staticmethod
    def from_dict(source_dict: dict[str, Any]) -> "Card":
        """Create a Card from a dictionary."""
        return Card(
            card_id=int(source_dict["card_id"]),
            state=State(int(source_dict["state"])),
            step=source_dict["step"],
            stability=float(source_dict["stability"])
            if source_dict["stability"]
            else None,
            difficulty=float(source_dict["difficulty"])
            if source_dict["difficulty"]
            else None,
            due=datetime.fromisoformat(source_dict["due"]),
            last_review=datetime.fromisoformat(source_dict["last_review"])
            if source_dict["last_review"]
            else None,
        )


class ReviewLog:
    """
    Log entry for a reviewed card.

    Attributes:
        card: Copy of the reviewed card.
        rating: Rating given during review.
        review_datetime: When the review occurred.
        review_duration: Milliseconds taken to review (optional).
    """

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
        """Convert review log to a JSON-serializable dictionary."""
        return {
            "card": self.card.to_dict(),
            "rating": self.rating.value,
            "review_datetime": self.review_datetime.isoformat(),
            "review_duration": self.review_duration,
        }

    @staticmethod
    def from_dict(source_dict: dict[str, Any]) -> "ReviewLog":
        """Create a ReviewLog from a dictionary."""
        return ReviewLog(
            card=Card.from_dict(source_dict["card"]),
            rating=Rating(int(source_dict["rating"])),
            review_datetime=datetime.fromisoformat(source_dict["review_datetime"]),
            review_duration=source_dict["review_duration"],
        )
