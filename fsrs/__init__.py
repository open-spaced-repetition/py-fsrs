"""
py-fsrs
-------

Py-FSRS is the official Python implementation of the FSRS scheduler algorithm, which can be used to develop spaced repetition systems.
"""

from fsrs.scheduler import Scheduler
from fsrs.state import State
from fsrs.card import Card
from fsrs.rating import Rating
from fsrs.review_log import ReviewLog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fsrs.optimizer import Optimizer


# lazy load the Optimizer module due to heavy dependencies
def __getattr__(name: str) -> type:
    if name == "Optimizer":
        global Optimizer
        from fsrs.optimizer import Optimizer

        return Optimizer
    raise AttributeError


__all__ = ["Scheduler", "Card", "Rating", "ReviewLog", "State", "Optimizer"]
