from copy import deepcopy
from datetime import timedelta

from fsrs.models import Card, Rating, SchedulingInfo, State
from fsrs.scheduler import AbstractScheduler


class BasicScheduler(AbstractScheduler):
    def new_state(self, rating: Rating) -> SchedulingInfo:
        if rating in self.next:
            return self.next[rating]

        next_card = deepcopy(self.current)
        next_card.difficulty = self.parameters.init_difficulty(rating)
        next_card.stability = self.parameters.init_stability(rating)

        if rating in (Rating.Again, Rating.Hard, Rating.Good):
            next_card.scheduled_days = 0
            next_card.state = State.Learning
            minutes = {Rating.Again: 1, Rating.Hard: 5, Rating.Good: 10}[rating]
            next_card.due = self.now + timedelta(minutes=minutes)
        elif rating == Rating.Easy:
            easy_interval = self.parameters.next_interval(
                next_card.stability,
                self.current.elapsed_days,
            )
            next_card.scheduled_days = easy_interval
            next_card.due = self.now + timedelta(days=easy_interval)
            next_card.state = State.Review
        else:
            raise ValueError("Invalid rating")

        scheduling_info = SchedulingInfo(
            card=next_card, review_log=self.build_log(rating)
        )
        self.next[rating] = scheduling_info
        return scheduling_info

    def learning_state(self, rating: Rating) -> SchedulingInfo:
        if rating in self.next:
            return self.next[rating]

        state, difficulty, stability = (
            self.last.state,
            self.last.difficulty,
            self.last.stability,
        )
        next_card = deepcopy(self.current)
        interval = self.current.elapsed_days
        next_card.difficulty = self.parameters.next_difficulty(difficulty, rating)
        next_card.stability = self.parameters.next_short_term_stability(
            stability, rating
        )

        if rating == Rating.Again:
            next_card.scheduled_days = 0
            next_card.due = self.now + timedelta(minutes=5)
            next_card.state = state
        elif rating == Rating.Hard:
            next_card.scheduled_days = 0
            next_card.due = self.now + timedelta(minutes=10)
            next_card.state = state
        elif rating == Rating.Good:
            good_interval = self.parameters.next_interval(next_card.stability, interval)
            next_card.scheduled_days = good_interval
            next_card.due = self.now + timedelta(days=good_interval)
            next_card.state = State.Review
        elif rating == Rating.Easy:
            good_stability = self.parameters.next_short_term_stability(
                stability, Rating.Good
            )
            good_interval = self.parameters.next_interval(good_stability, interval)
            easy_interval = max(
                self.parameters.next_interval(next_card.stability, interval),
                good_interval + 1,
            )
            next_card.scheduled_days = easy_interval
            next_card.due = self.now + timedelta(days=easy_interval)
            next_card.state = State.Review
        else:
            raise ValueError("Invalid rating")

        scheduling_info = SchedulingInfo(
            card=next_card, review_log=self.build_log(rating)
        )
        self.next[rating] = scheduling_info
        return scheduling_info

    def review_state(self, rating: Rating) -> SchedulingInfo:
        if rating in self.next:
            return self.next[rating]

        interval = self.current.elapsed_days
        difficulty, stability = self.last.difficulty, self.last.stability
        retrievability = self.parameters.forgetting_curve(interval, stability)
        next_again = deepcopy(self.current)
        next_hard = deepcopy(self.current)
        next_good = deepcopy(self.current)
        next_easy = deepcopy(self.current)

        self._next_ds(
            next_again,
            next_hard,
            next_good,
            next_easy,
            difficulty,
            stability,
            retrievability,
        )
        self._next_interval(next_again, next_hard, next_good, next_easy, interval)
        self._next_state(next_again, next_hard, next_good, next_easy)
        next_again.lapses += 1

        self.next[Rating.Again] = SchedulingInfo(
            card=next_again, review_log=self.build_log(Rating.Again)
        )
        self.next[Rating.Hard] = SchedulingInfo(
            card=next_hard, review_log=self.build_log(Rating.Hard)
        )
        self.next[Rating.Good] = SchedulingInfo(
            card=next_good, review_log=self.build_log(Rating.Good)
        )
        self.next[Rating.Easy] = SchedulingInfo(
            card=next_easy, review_log=self.build_log(Rating.Easy)
        )

        return self.next[rating]

    def _next_ds(
        self,
        next_again: Card,
        next_hard: Card,
        next_good: Card,
        next_easy: Card,
        difficulty: float,
        stability: float,
        retrievability: float,
    ) -> None:
        next_again.difficulty = self.parameters.next_difficulty(
            difficulty, Rating.Again
        )
        next_again.stability = self.parameters.next_forget_stability(
            difficulty, stability, retrievability
        )
        next_hard.difficulty = self.parameters.next_difficulty(difficulty, Rating.Hard)
        next_hard.stability = self.parameters.next_recall_stability(
            difficulty, stability, retrievability, Rating.Hard
        )
        next_good.difficulty = self.parameters.next_difficulty(difficulty, Rating.Good)
        next_good.stability = self.parameters.next_recall_stability(
            difficulty, stability, retrievability, Rating.Good
        )
        next_easy.difficulty = self.parameters.next_difficulty(difficulty, Rating.Easy)
        next_easy.stability = self.parameters.next_recall_stability(
            difficulty, stability, retrievability, Rating.Easy
        )

    def _next_interval(
        self,
        next_again: Card,
        next_hard: Card,
        next_good: Card,
        next_easy: Card,
        interval: int,
    ) -> None:
        hard_interval = self.parameters.next_interval(next_hard.stability, interval)
        good_interval = self.parameters.next_interval(next_good.stability, interval)
        hard_interval = min(hard_interval, good_interval)
        good_interval = max(good_interval, hard_interval + 1)
        easy_interval = max(
            self.parameters.next_interval(next_easy.stability, interval),
            good_interval + 1,
        )

        next_again.scheduled_days = 0
        next_again.due = self.now + timedelta(minutes=5)

        next_hard.scheduled_days = hard_interval
        next_hard.due = self.now + timedelta(days=hard_interval)
        next_good.scheduled_days = good_interval
        next_good.due = self.now + timedelta(days=good_interval)
        next_easy.scheduled_days = easy_interval
        next_easy.due = self.now + timedelta(days=easy_interval)

    def _next_state(
        self, next_again: Card, next_hard: Card, next_good: Card, next_easy: Card
    ) -> None:
        next_again.state = State.Relearning
        next_hard.state = State.Review
        next_good.state = State.Review
        next_easy.state = State.Review
