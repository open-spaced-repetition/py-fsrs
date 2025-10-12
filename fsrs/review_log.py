"""
fsrs.review_log
---------

This module defines the ReviewLog and Rating classes.

Classes:
    ReviewLog: Represents the log entry of a Card that has been reviewed.
    Rating: Enum representing the four possible ratings when reviewing a card.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict
import json
from typing_extensions import Self
from fsrs.rating import Rating


class ReviewLogDict(TypedDict):
    """
    JSON-serializable dictionary representation of a ReviewLog object.
    """

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
        Returns a dictionary representation of the ReviewLog object.

        Returns:
            A dictionary representation of the ReviewLog object.
        """

        return {
            "card_id": self.card_id,
            "rating": int(self.rating),
            "review_datetime": self.review_datetime.isoformat(),
            "review_duration": self.review_duration,
        }

    @classmethod
    def from_dict(
        cls,
        source_dict: ReviewLogDict,
    ) -> Self:
        """
        Creates a ReviewLog object from an existing dictionary.

        Args:
            source_dict: A dictionary representing an existing ReviewLog object.

        Returns:
            A ReviewLog object created from the provided dictionary.
        """

        return cls(
            card_id=source_dict["card_id"],
            rating=Rating(int(source_dict["rating"])),
            review_datetime=datetime.fromisoformat(source_dict["review_datetime"]),
            review_duration=source_dict["review_duration"],
        )

    def to_json(self, indent: int | str | None = None) -> str:
        """
        Returns a JSON-serialized string of the ReviewLog object.

        Args:
            indent: Equivalent argument to the indent in json.dumps()

        Returns:
            str: A JSON-serialized string of the ReviewLog object.
        """

        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, source_json: str) -> Self:
        """
        Creates a ReviewLog object from a JSON-serialized string.

        Args:
            source_json: A JSON-serialized string of an existing ReviewLog object.

        Returns:
            Self: A ReviewLog object created from the JSON string.
        """

        source_dict: ReviewLogDict = json.loads(source_json)
        return cls.from_dict(source_dict=source_dict)


__all__ = ["ReviewLog"]
