"""
fsrs.fsrs
---------

This module defines the FSRS scheduler.

Classes:
    FSRS: The FSRS scheduler.
"""

from fsrs.scheduler_basic import BasicScheduler
from fsrs.scheduler_long_term import LongTermScheduler
from .models import (
    Card,
    Rating,
    SchedulingInfo,
    Parameters,
)
from datetime import datetime, timezone
from typing import Optional


class FSRS:
    """
    The FSRS scheduler.

    Enables the reviewing and future scheduling of cards according to the FSRS algorithm.

    Attributes:
        p (Parameters): Object for configuring the scheduler's model weights, desired retention and maximum interval.
    """

    p: Parameters

    def __init__(
        self,
        parameters: Optional[Parameters] = None,
    ) -> None:
        """
        Initializes the FSRS scheduler.

        Args:
            w (Optional[tuple[float, ...]]): The 19 model weights of the FSRS scheduler.
            request_retention (Optional[float]): The desired retention of the scheduler. Corresponds to the maximum retrievability a Card object can have before it is due.
            maximum_interval (Optional[int]): The maximum number of days into the future a Card object can be scheduled for next review.
        """
        self.p = parameters if parameters else Parameters()
        self.Scheduler = (
            BasicScheduler if self.p.enable_short_term else LongTermScheduler
        )

    def next(
        self, card: Card, rating: Rating, now: Optional[datetime] = None
    ) -> SchedulingInfo:
        """
        Reviews a card for a given rating.

        Args:
            card (Card): The card being reviewed.
            rating (Rating): The chosen rating for the card being reviewed.
            now (Optional[datetime]): The date and time of the review.

        Returns:
            tuple: A tuple containing the updated, reviewed card and its corresponding review log.

        Raises:
            ValueError: If the `now` argument is not timezone-aware and set to UTC.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        if (now.tzinfo is None) or (now.tzinfo != timezone.utc):
            raise ValueError("datetime must be timezone-aware and set to UTC")
        return self.Scheduler(card, now, self.p).review(rating)

    def repeat(
        self, card: Card, now: Optional[datetime] = None
    ) -> dict[Rating, SchedulingInfo]:
        if now is None:
            now = datetime.now(timezone.utc)
        if (now.tzinfo is None) or (now.tzinfo != timezone.utc):
            raise ValueError("datetime must be timezone-aware and set to UTC")
        return self.Scheduler(card, now, self.p).preview().log_items
