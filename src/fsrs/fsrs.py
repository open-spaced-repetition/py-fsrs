from .models import *
import math


class FSRS:
    p: Parameters

    def __init__(self) -> None:
        self.p = Parameters()

    def repeat(self, card: Card, now: datetime) -> dict[int, SchedulingInfo]:
        if card.state == State.New:
            card.elapsed_days = 0
        else:
            card.elapsed_days = (now - card.last_review).days
        card.last_review = now
        card.reps += 1
        s = SchedulingCards(card)
        s.update_state(card.state)

        if card.state == State.New:
            self.init_ds(s)

            s.again.due = now + timedelta(minutes=1)
            s.hard.due = now + timedelta(minutes=5)
            s.good.due = now + timedelta(minutes=10)
            easy_interval = self.next_interval(s.easy.stability * self.p.easy_bonus)
            s.easy.scheduled_days = easy_interval
            s.easy.due = now + timedelta(days=easy_interval)
        elif card.state == State.Learning or card.state == State.Relearning:
            hard_interval = self.next_interval(s.hard.stability)
            good_interval = max(self.next_interval(s.good.stability), hard_interval + 1)
            easy_interval = max(self.next_interval(s.easy.stability * self.p.easy_bonus), good_interval + 1)

            s.schedule(now, hard_interval, good_interval, easy_interval)
        elif card.state == State.Review:
            interval = card.elapsed_days
            last_d = card.difficulty
            last_s = card.stability
            retrievability = math.exp(math.log(0.9) * interval / last_s)
            self.next_ds(s, last_d, last_s, retrievability)

            hard_interval = self.next_interval(last_s * self.p.hard_factor)
            good_interval = self.next_interval(s.good.stability)
            hard_interval = min(hard_interval, good_interval)
            good_interval = max(good_interval, hard_interval + 1)
            easy_interval = max(self.next_interval(s.easy.stability * self.p.hard_factor), good_interval + 1)
            s.schedule(now, hard_interval, good_interval, easy_interval)
        return s.record_log(card, now)

    def init_ds(self, s: SchedulingCards) -> None:
        s.again.difficulty = self.init_difficulty(Rating.Again)
        s.again.stability = self.init_stability(Rating.Again)
        s.hard.difficulty = self.init_difficulty(Rating.Hard)
        s.hard.stability = self.init_stability(Rating.Hard)
        s.good.difficulty = self.init_difficulty(Rating.Good)
        s.good.stability = self.init_stability(Rating.Good)
        s.easy.difficulty = self.init_difficulty(Rating.Easy)
        s.easy.stability = self.init_stability(Rating.Easy)

    def next_ds(self, s: SchedulingCards, last_d: float, last_s: float, retrievability: float):
        s.again.difficulty = self.next_difficulty(last_d, Rating.Again)
        s.again.stability = self.next_forget_stability(s.again.difficulty, last_s, retrievability)
        s.hard.difficulty = self.next_difficulty(last_d, Rating.Hard)
        s.hard.stability = self.next_recall_stability(s.hard.difficulty, last_s, retrievability)
        s.good.difficulty = self.next_difficulty(last_d, Rating.Good)
        s.good.stability = self.next_recall_stability(s.good.difficulty, last_s, retrievability)
        s.easy.difficulty = self.next_difficulty(last_d, Rating.Easy)
        s.easy.stability = self.next_recall_stability(s.easy.difficulty, last_s, retrievability)

    def init_stability(self, r: int) -> float:
        return max(self.p.w[0] + self.p.w[1] * r, 0.1)

    def init_difficulty(self, r: int) -> float:
        return min(max(self.p.w[2] + self.p.w[3] * (r - 2), 1), 10)

    def next_interval(self, s: float) -> int:
        State.New_interval = s * math.log(self.p.request_retention) / math.log(0.9)
        return min(max(round(State.New_interval), 1), self.p.maximum_interval)

    def next_difficulty(self, d: float, r: int) -> float:
        next_d = d + self.p.w[4] * (r - 2)
        return min(max(self.mean_reversion(self.p.w[2], next_d), 1), 10)

    def mean_reversion(self, init: float, current: float) -> float:
        return self.p.w[5] * init + (1 - self.p.w[5]) * current

    def next_recall_stability(self, d: float, s: float, r: float) -> float:
        return s * (1 + math.exp(self.p.w[6]) *
                    (11 - d) *
                    math.pow(s, self.p.w[7]) *
                    (math.exp((1 - r) * self.p.w[8]) - 1))

    def next_forget_stability(self, d: float, s: float, r: float) -> float:
        return self.p.w[9] * math.pow(d, self.p.w[10]) * math.pow(s, self.p.w[11]) * math.exp((1 - r) * self.p.w[12])
