"""
py-fsrs
-------

Py-FSRS is the official Python implementation of the FSRS scheduler algorithm, which can be used to develop spaced repetition systems.
"""

from fsrs.scheduler import Scheduler
from fsrs.card import Card, State
from fsrs.review_log import ReviewLog, Rating


# lazy load the Optimizer module due to heavy dependencies
def __getattr__(name):
    if name == "Optimizer":
        global Optimizer
        from fsrs.optimizer import Optimizer

        return Optimizer
    raise AttributeError


__all__ = ["Scheduler", "Card", "Rating", "ReviewLog", "State", "Optimizer"]
