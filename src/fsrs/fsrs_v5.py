from datetime import datetime
from math import exp

from fsrs.models import Rating


class FSRSv5:
    """
    Free Spaced Repetition Scheduler v5 implementation.

    This class implements the FSRS v5 algorithm for spaced repetition scheduling.
    It calculates stability and difficulty values for flashcards based on review ratings.

    The algorithm uses the following key concepts:
    - Retrievability (R): Probability of recall at a given time, calculated using a power function
      that better models the superposition of multiple exponential forgetting curves
    - Stability (S): Memory strength, measured as the time interval when R=0.9. Stability increases
      are larger for easier cards, newer memories, and when reviewing at lower retrievability
    - Difficulty (D): Value between 1-10 indicating card difficulty. Higher difficulty results in
      smaller stability increases. Difficulty changes based on ratings and reverts toward an optimal value
    - Rating (G): Review rating (Again=1, Hard=2, Good=3, Easy=4)

    The scheduler uses 19 model weights (w0-w18) to calculate:
    - Initial difficulty and stability for new cards (w0-w5)
    - Stability changes after successful/failed reviews (w6-w16)
    - Difficulty adjustments based on ratings
    - Short-term stability changes for same-day reviews (w17-w18)
    """

    def __init__(self, parameters: tuple[float, ...] | list[float]):
        self.w = parameters
        # Power law decay rate fixed at -0.5 for optimal forgetting curve fit
        self.decay = -0.5
        # Factor derived to ensure R=0.9 when t=S
        self.factor = 0.9 ** (1 / self.decay) - 1

    def next_stability_and_difficulty(
        self,
        stability: float | None,
        difficulty: float | None,
        review_at: datetime,
        last_review_at: datetime | None,
        rating: Rating,
    ) -> tuple[float, float]:
        """
        Calculate the next stability and difficulty values after a review.

        For first reviews, uses initial stability/difficulty formulas.
        For subsequent reviews, calculates retrievability and updates both values
        based on the rating and time elapsed.

        Args:
            stability: Current stability value or None if first review
            difficulty: Current difficulty value or None if first review
            review_at: When the review occurred
            last_review_at: When the previous review occurred, or None if first review
            rating: The rating given during review

        Returns:
            Tuple of (new stability, new difficulty)
        """
        if stability is None or difficulty is None:
            return self._initial_stability(rating), self._initial_difficulty(rating)
        S, D, t, last_t, G = stability, difficulty, review_at, last_review_at, rating
        return self._next_stability(S, D, t, last_t, G), self._next_difficulty(D, G)

    def interval(self, stability: float, desired_retention: float) -> float:
        """
        Calculate the next interval given stability and desired retention.

        Uses the power function formula: I(r,S) = (S/FACTOR) * (r^(1/DECAY) - 1)
        This provides optimal spacing by targeting the desired retention probability.

        Args:
            stability: Current stability value
            desired_retention: Target retention probability

        Returns:
            Next interval in days
        """
        return (stability / self.factor) * ((desired_retention ** (1 / self.decay)) - 1)

    def _next_stability(
        self, S: float, D: float, t: datetime, last_t: datetime | None, G: Rating
    ) -> float:
        """
        Calculate the next stability value based on current state and rating.

        For same-day reviews: S' = S * e^(w17 * (G-3 + w18))
        For regular reviews:
        - If Again: Uses minimum of forget and short-term stability formulas
        - If Hard/Good/Easy: Uses recall stability formula that factors in:
          - Higher D -> smaller increase
          - Higher S -> harder to increase further
          - Lower R -> larger increase (spacing effect)

        Args:
            S: Current stability
            D: Current difficulty
            t: Review time
            last_t: Last review time
            G: Rating given

        Returns:
            New stability value
        """
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
        """
        Calculate retrievability (probability of recall) after time interval.

        Uses power function: R(t,S) = (1 + FACTOR*t/S)^DECAY
        This provides better fit than exponential, especially for mixed memory strengths.

        Args:
            S: Current stability
            t: Current time
            last_t: Last review time

        Returns:
            Retrievability value between 0-1
        """
        if last_t is None:
            return 0
        return (1 + self.factor * max(0, (t - last_t).days) / S) ** self.decay

    def _initial_stability(self, G: Rating) -> float:
        """
        Get initial stability value based on first rating.

        Uses weights w0-w3 for ratings 1-4 respectively.
        These weights are first estimated by curve fitting, then fine-tuned by gradient descent.

        Args:
            G: Rating given

        Returns:
            Initial stability value
        """
        w0, w1, w2, w3 = self.w[0], self.w[1], self.w[2], self.w[3]
        return {Rating.Again: w0, Rating.Hard: w1, Rating.Good: w2, Rating.Easy: w3}[G]

    def _initial_difficulty(self, G: Rating) -> float:
        """
        Calculate initial difficulty based on first rating.

        Uses formula: D0(G) = w4 - e^(w5*(G-1)) + 1
        This is a change from FSRS 4.5 that provides better fit.
        Values <1 are clamped to 1, values >10 are clamped to 10.

        Args:
            G: Rating given

        Returns:
            Initial difficulty value between 1-10
        """
        return self.w[4] - exp(self.w[5] * (G - 1)) + 1

    def _next_difficulty(self, D: float, G: Rating) -> float:
        """
        Calculate next difficulty value after a review.

        Uses formulas:
        ΔD = -w6*(G-3)
        D' = D + ΔD*(10-D)/9  # Linear damping term
        D'' = w7*D0(4) + (1-w7)*D'  # Mean reversion to Easy initial difficulty

        The (10-D)/9 term makes difficulty approach 10 asymptotically.
        Mean reversion uses unclamped D0(4) value.

        Args:
            D: Current difficulty
            G: Rating given

        Returns:
            New difficulty value between 1-10
        """
        D04 = self._initial_difficulty(Rating.Easy)
        delta_D = -self.w[6] * (G - 3)
        D_prime = D + delta_D * (10 - D) / 9
        D_double_prime = self.w[7] * D04 + (1 - self.w[7]) * D_prime
        return max(min(D_double_prime, 10), 1)

    def _recall_stability(self, S: float, D: float, R: float, rating: Rating) -> float:
        """
        Calculate new stability after successful recall (Hard, Good or Easy ratings).

        Formula: S'_r(D,S,R,G) = S * (e^w8 * (11-D) * S^-w9 * (e^(w10*(1-R)) - 1) * w15^(if G=2) * w16^(if G=4) + 1)

        Key effects:
        - Higher D -> smaller increase (linear: 11-D)
        - Higher S -> harder to increase (power law: S^-w9)
        - Lower R -> larger increase (exponential: e^(w10*(1-R)))
        - Hard rating applies penalty w15
        - Easy rating applies bonus w16

        Args:
            S: Current stability
            D: Current difficulty
            R: Current retrievability
            rating: Rating given (Hard, Good or Easy)

        Returns:
            New stability value after successful recall
        """
        w8, w9, w10 = self.w[8], self.w[9], self.w[10]
        factor = self._hard_penalty(rating) * self._easy_bonus(rating)
        return S * (
            1 + exp(w8) * (11 - D) * pow(S, -w9) * (exp((1 - R) * w10) - 1) * factor
        )

    def _long_term_forget_stability(self, S: float, D: float, R: float) -> float:
        """
        Calculate new stability after forgetting (post-lapse stability).

        Formula: S'_f(D,S,R) = w11 * D^(-w12) * ((S+1)^w13 - 1) * e^(w14*(1-R))
        Result is capped at current stability S.

        Uses nonlinear difficulty function D^(-w12) unlike recall stability.

        Args:
            S: Current stability
            D: Current difficulty
            R: Current retrievability

        Returns:
            New stability value after forgetting
        """
        w11, w12, w13, w14 = self.w[11], self.w[12], self.w[13], self.w[14]
        return w11 * pow(D, -w12) * (pow(S + 1, w13) - 1) * exp((1 - R) * w14)

    def _hard_penalty(self, rating: Rating) -> float:
        """Apply penalty multiplier for Hard rating (0 < w15 < 1)."""
        return self.w[15] if rating == Rating.Hard else 1

    def _easy_bonus(self, rating: Rating) -> float:
        """Apply bonus multiplier for Easy rating (1 < w16 < 6)."""
        return self.w[16] if rating == Rating.Easy else 1

    def _short_term_stability(self, S: float, G: float) -> float:
        """
        Calculate stability for same-day reviews.

        Uses formula: S' = S * e^(w17 * (G-3 + w18))
        This is new in FSRS 5 to handle same-day reviews when exact intervals unknown.

        Args:
            S: Current stability
            G: Rating given

        Returns:
            New stability value
        """
        return S * exp(self.w[17] * (G - 3 + self.w[18]))
