from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Dict, Iterator
from fsrs.models import Card, Parameters, Rating, ReviewLog, SchedulingInfo, State


class IPreview:
    log_items: Dict[Rating, SchedulingInfo]

    def __init__(self):
        self.log_items = {}

    def __iter__(self) -> Iterator[SchedulingInfo]:
        return iter(self.log_items.values())


class IScheduler(ABC):
    @abstractmethod
    def preview(self) -> IPreview:
        pass

    @abstractmethod
    def review(self, rating: Rating) -> SchedulingInfo:
        pass


class AbstractScheduler(IScheduler):
    def __init__(self, card: Card, now: datetime, parameters: Parameters):
        self.parameters = parameters
        self.last = deepcopy(card)
        self.current = deepcopy(card)
        self.now = now
        self.next: Dict[Rating, SchedulingInfo] = {}
        self._init()

    def _init(self):
        state, last_review = self.current.state, self.current.last_review
        interval = 0
        if state != State.New and last_review:
            interval = (self.now - last_review).days
        self.current.last_review = self.now
        self.current.elapsed_days = interval
        self.current.reps += 1
        self._init_seed()

    def preview(self) -> IPreview:
        preview = IPreview()
        for rating in Rating:
            preview.log_items[rating] = self.review(rating)
        return preview

    def review(self, rating: Rating) -> SchedulingInfo:
        state = self.last.state
        if state == State.New:
            return self.new_state(rating)
        elif state in (State.Learning, State.Relearning):
            return self.learning_state(rating)
        elif state == State.Review:
            return self.review_state(rating)
        raise ValueError("Invalid rating")

    @abstractmethod
    def new_state(self, rating: Rating) -> SchedulingInfo:
        pass

    @abstractmethod
    def learning_state(self, rating: Rating) -> SchedulingInfo:
        pass

    @abstractmethod
    def review_state(self, rating: Rating) -> SchedulingInfo:
        pass

    def build_log(self, rating: Rating) -> ReviewLog:
        log = ReviewLog(
            rating=rating,
            scheduled_days=self.current.scheduled_days,
            elapsed_days=self.current.elapsed_days,
            review=self.now,
            state=self.current.state,
        )
        return log
