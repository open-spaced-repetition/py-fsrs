from datetime import datetime
from math import exp

from fsrs.models import Rating


class FSRSv5:
    def __init__(
        self,
        parameters: tuple[float, ...] | list[float],
        decay: float = -0.5,
    ):
        self.w = parameters
        self.factor = 0.9 ** (1 / decay) - 1
        self.decay = decay

    def next_stability_and_difficulty(
        self,
        stability: float | None,
        difficulty: float | None,
        review_at: datetime,
        last_review_at: datetime | None,
        rating: Rating,
    ) -> tuple[float, float]:
        if stability is None or difficulty is None:
            return self._initial_stability(rating), self._initial_difficulty(rating)
        S, D, t, last_t, G = stability, difficulty, review_at, last_review_at, rating
        return self._next_stability(S, D, t, last_t, G), self._next_difficulty(D, G)

    def interval(self, stability: float, desired_retention: float) -> float:
        return (stability / self.factor) * ((desired_retention ** (1 / self.decay)) - 1)

    def _next_stability(
        self, S: float, D: float, t: datetime, last_t: datetime | None, G: Rating
    ) -> float:
        R = self._retrievability(S, t, last_t)
        if last_t and (t - last_t).days < 1:
            return self._short_term_stability(S, G)
        elif G == Rating.Again:
            return min(
                self._long_term_forget_stability(S, D, R),
                self._short_term_stability(S, G),
            )
        else:
            return self._recall_stability(S, D, R, G)

    def _retrievability(self, S: float, t: datetime, last_t: datetime | None) -> float:
        if last_t is None:
            return 0
        return (1 + self.factor * max(0, (t - last_t).days) / S) ** self.decay

    def _initial_stability(self, G: Rating) -> float:
        w0, w1, w2, w3 = self.w[0], self.w[1], self.w[2], self.w[3]
        return {Rating.Again: w0, Rating.Hard: w1, Rating.Good: w2, Rating.Easy: w3}[G]

    def _initial_difficulty(self, G: Rating) -> float:
        return self.w[4] - exp(self.w[5] * (G - 1)) + 1

    def _next_difficulty(self, D: float, G: Rating) -> float:
        D04 = self._initial_difficulty(Rating.Easy)
        delta_D = -self.w[6] * (G - 3)
        D_prime = D + delta_D * (10 - D) / 9
        D_double_prime = self.w[7] * D04 + (1 - self.w[7]) * D_prime
        return max(min(D_double_prime, 10), 1)

    def _recall_stability(self, S: float, D: float, R: float, rating: Rating) -> float:
        w8, w9, w10 = self.w[8], self.w[9], self.w[10]
        factor = self._hard_penalty(rating) * self._easy_bonus(rating)
        return S * (
            1 + exp(w8) * (11 - D) * pow(S, -w9) * (exp((1 - R) * w10) - 1) * factor
        )

    def _long_term_forget_stability(self, S: float, D: float, R: float) -> float:
        w11, w12, w13, w14 = self.w[11], self.w[12], self.w[13], self.w[14]
        return w11 * pow(D, -w12) * (pow(S + 1, w13) - 1) * exp((1 - R) * w14)

    def _hard_penalty(self, rating: Rating) -> float:
        return self.w[15] if rating == Rating.Hard else 1

    def _easy_bonus(self, rating: Rating) -> float:
        return self.w[16] if rating == Rating.Easy else 1

    def _short_term_stability(self, S: float, G: float) -> float:
        return S * exp(self.w[17] * (G - 3 + self.w[18]))
