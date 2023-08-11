import math
from src.fsrs.models import *
from datetime import timedelta


class FSRS:
    params: Parameters

    def __init__(self) -> None:
        self.params = Parameters()

    def review(self, card: Card, rating: Rating, now=datetime.utcnow()):
        card.reps += 1

        if card.state == State.New:
            card.elapsed_days = 0
        else:
            card.elapsed_days = (now - card.last_review).days
        card.last_review = now

        match (card.state):
            case State.New:
                card.stability = self.init_stability(rating)
                card.difficulty = self.init_difficulty(rating)

                match (rating):
                    case Rating.Again:
                        card.due = now + timedelta(minutes=1)
                    case Rating.Hard:
                        card.due = now + timedelta(minutes=5)
                    case Rating.Good:
                        card.due = now + timedelta(minutes=10)
                    case Rating.Easy:
                        easy_interval = self.next_interval(card.stability)
                        card.scheduled_days = easy_interval
                        card.due = now + timedelta(days=easy_interval)

            case State.Learning | State.Relearning:
                match (rating):
                    case Rating.Again:
                        card.scheduled_days = 0
                        card.due = now + timedelta(minutes=5)
                    case Rating.Hard:
                        card.scheduled_days = 0
                        card.due = now + timedelta(minutes=10)
                    case Rating.Good:
                        interval = self.next_interval(card.stability)
                        card.scheduled_days = interval
                        card.due = now + timedelta(days=interval)
                    case Rating.Easy:
                        good_interval = self.next_interval(card.stability)
                        interval = max(
                            self.next_interval(card.stability), good_interval + 1
                        )
                        card.scheduled_days = interval
                        card.due = now + timedelta(days=interval)

            case State.Review:
                retrievability = card.get_retrievability()
                card.difficulty = self.next_difficulty(card.difficulty, rating)

                if rating == Rating.Again:
                    card.stability = self.next_forget_stability(
                        card.difficulty, card.stability, retrievability
                    )
                    card.scheduled_days = 0
                    card.due = now + timedelta(minutes=5)
                else:
                    card.stability = self.next_recall_stability(
                        card.difficulty, card.stability, retrievability, rating
                    )
                    interval = self.next_interval(card.stability)
                    card.scheduled_days = interval
                    card.due = now + timedelta(days=interval)

        card.update_state(rating)
        return card

    def init_stability(self, r: int) -> float:
        return max(self.params.w[r - 1], 0.1)

    def init_difficulty(self, r: int) -> float:
        return min(max(self.params.w[4] - self.params.w[5] * (r - 3), 1), 10)

    def mean_reversion(self, init: float, current: float) -> float:
        return self.params.w[7] * init + (1 - self.params.w[7]) * current

    def next_interval(self, s: float) -> int:
        new_interval = s * 9 * (1 / self.params.request_retention - 1)
        return min(max(round(new_interval), 1), self.params.maximum_interval)

    def next_difficulty(self, d: float, r: int) -> float:
        next_d = d - self.params.w[6] * (r - 3)
        return min(max(self.mean_reversion(self.params.w[4], next_d), 1), 10)

    def next_recall_stability(self, d: float, s: float, r: float, rating: int) -> float:
        hard_penalty = self.params.w[15] if rating == Rating.Hard else 1
        easy_bonus = self.params.w[16] if rating == Rating.Easy else 1
        return s * (
            1
            + math.exp(self.params.w[8])
            * (11 - d)
            * math.pow(s, -self.params.w[9])
            * (math.exp((1 - r) * self.params.w[10]) - 1)
            * hard_penalty
            * easy_bonus
        )

    def next_forget_stability(self, d: float, s: float, r: float) -> float:
        return (
            self.params.w[11]
            * math.pow(d, -self.params.w[12])
            * (math.pow(s + 1, self.params.w[13]) - 1)
            * math.exp((1 - r) * self.params.w[14])
        )
