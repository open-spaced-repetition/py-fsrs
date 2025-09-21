"""
fsrs.review_log
---------

This module defines the ReviewLog and Rating classes.

Classes:
    ReviewLog: Represents the log entry of a Card that has been reviewed.
    Rating: Enum representing the four possible ratings when reviewing a card.
"""

from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict


class Rating(IntEnum):
    """
    Enum representing the four possible ratings when reviewing a card.
    """

    Again = 1
    Hard = 2
    Good = 3
    Easy = 4


class ReviewLogDict(TypedDict):
    card_id: int
    rating: int
    review_datetime: str
    review_duration: int | None


@dataclass
class ReviewLog:
    """
    Represents the log entry of a Card object that has been reviewed.

    Attributes:
        card_id: The id of the card being reviewed.
        rating: The rating given to the card during the review.
        review_datetime: The date and time of the review.
        review_duration: The number of milliseconds it took to review the card or None if unspecified.
    """

    card_id: int
    rating: Rating
    review_datetime: datetime
    review_duration: int | None

    def to_dict(
        self,
    ) -> ReviewLogDict:
        """
        Returns a JSON-serializable dictionary representation of the ReviewLog object.

        This method is specifically useful for storing ReviewLog objects in a database.

        Returns:
            A dictionary representation of the ReviewLog object.
        """

        return {
            "card_id": self.card_id,
            "rating": int(self.rating),
            "review_datetime": self.review_datetime.isoformat(),
            "review_duration": self.review_duration,
        }

    @staticmethod
    def from_dict(
        source_dict: ReviewLogDict,
    ) -> ReviewLog:
        """
        Creates a ReviewLog object from an existing dictionary.

        Args:
            source_dict: A dictionary representing an existing ReviewLog object.

        Returns:
            A ReviewLog object created from the provided dictionary.
        """

        return ReviewLog(
            card_id=source_dict["card_id"],
            rating=Rating(int(source_dict["rating"])),
            review_datetime=datetime.fromisoformat(source_dict["review_datetime"]),
            review_duration=source_dict["review_duration"],
        )


__all__ = ["ReviewLog", "Rating"]
