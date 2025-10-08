"""
fsrs.card
---------

This module defines the Card and State classes.

Classes:
    Card: Represents a flashcard in the FSRS system.
    State: Enum representing the learning state of a Card object.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import TypedDict
from typing_extensions import Self
from fsrs.state import State


class CardDict(TypedDict):
    """
    JSON-serializable dictionary representation of a Card object.
    """

    card_id: int
    state: int
    step: int | None
    stability: float | None
    difficulty: float | None
    due: str
    last_review: str | None


@dataclass(init=False)
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

    def to_dict(self) -> CardDict:
        """
        Returns a JSON-serializable dictionary representation of the Card object.

        This method is specifically useful for storing Card objects in a database.

        Returns:
            A dictionary representation of the Card object.
        """

        return {
            "card_id": self.card_id,
            "state": self.state.value,
            "step": self.step,
            "stability": self.stability,
            "difficulty": self.difficulty,
            "due": self.due.isoformat(),
            "last_review": self.last_review.isoformat() if self.last_review else None,
        }

    @classmethod
    def from_dict(cls, source_dict: CardDict) -> Self:
        """
        Creates a Card object from an existing dictionary.

        Args:
            source_dict: A dictionary representing an existing Card object.

        Returns:
            A Card object created from the provided dictionary.
        """

        return cls(
            card_id=int(source_dict["card_id"]),
            state=State(int(source_dict["state"])),
            step=source_dict["step"],
            stability=(
                float(source_dict["stability"]) if source_dict["stability"] else None
            ),
            difficulty=(
                float(source_dict["difficulty"]) if source_dict["difficulty"] else None
            ),
            due=datetime.fromisoformat(source_dict["due"]),
            last_review=(
                datetime.fromisoformat(source_dict["last_review"])
                if source_dict["last_review"]
                else None
            ),
        )


__all__ = ["Card"]
