from copy import deepcopy
from datetime import timedelta

from fsrs.models import Card, Rating, SchedulingInfo, State
from fsrs.scheduler import AbstractScheduler


class LongTermScheduler(AbstractScheduler):
    def new_state(self, rating: Rating) -> SchedulingInfo:
        if rating in self.next:
            return self.next[rating]

        self.current.scheduled_days = 0
        self.current.elapsed_days = 0

        next_again = deepcopy(self.current)
        next_hard = deepcopy(self.current)
        next_good = deepcopy(self.current)
        next_easy = deepcopy(self.current)

        self._init_ds(next_again, next_hard, next_good, next_easy)
        first_interval = 0

        self._next_interval(next_again, next_hard, next_good, next_easy, first_interval)
        self._next_state(next_again, next_hard, next_good, next_easy)
        self._update_next(next_again, next_hard, next_good, next_easy)
        return self.next[rating]

    def _init_ds(
        self, next_again: Card, next_hard: Card, next_good: Card, next_easy: Card
    ) -> None:
        next_again.difficulty = self.parameters.init_difficulty(Rating.Again)
        next_again.stability = self.parameters.init_stability(Rating.Again)

        next_hard.difficulty = self.parameters.init_difficulty(Rating.Hard)
        next_hard.stability = self.parameters.init_stability(Rating.Hard)

        next_good.difficulty = self.parameters.init_difficulty(Rating.Good)
        next_good.stability = self.parameters.init_stability(Rating.Good)

        next_easy.difficulty = self.parameters.init_difficulty(Rating.Easy)
        next_easy.stability = self.parameters.init_stability(Rating.Easy)

    def learning_state(self, rating: Rating) -> SchedulingInfo:
        return self.review_state(rating)

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

        self._update_next(next_again, next_hard, next_good, next_easy)
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
        again_interval = self.parameters.next_interval(next_again.stability, interval)
        hard_interval = self.parameters.next_interval(next_hard.stability, interval)
        good_interval = self.parameters.next_interval(next_good.stability, interval)
        easy_interval = self.parameters.next_interval(next_easy.stability, interval)

        again_interval = min(again_interval, hard_interval)
        hard_interval = max(hard_interval, again_interval + 1)
        good_interval = max(good_interval, hard_interval + 1)
        easy_interval = max(easy_interval, good_interval + 1)

        for card, interval in zip(
            [next_again, next_hard, next_good, next_easy],
            [again_interval, hard_interval, good_interval, easy_interval],
        ):
            card.scheduled_days = interval
            card.due = self.now + timedelta(days=interval)

    def _next_state(
        self, next_again: Card, next_hard: Card, next_good: Card, next_easy: Card
    ) -> None:
        for card in (next_again, next_hard, next_good, next_easy):
            card.state = State.Review

    def _update_next(
        self, next_again: Card, next_hard: Card, next_good: Card, next_easy: Card
    ) -> None:
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
