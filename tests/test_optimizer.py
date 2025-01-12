from fsrs import Scheduler, Card, ReviewLog, State, Rating, Optimizer
import pytest

starting_parameters = [
            0.40255,
            1.18385,
            3.173,
            15.69105,
            7.1949,
            0.5345,
            1.4604,
            0.0046,
            1.54575,
            0.1192,
            1.01925,
            1.9395,
            0.11,
            0.29605,
            2.2698,
            0.2315,
            2.9898,
            0.51655,
            0.6621,
]

class TestOptimizer:

    def test_zero_revlogs(self):

        # if no review logs are provided, the optimal parameters should not change
        # from the starting parameters

        review_logs = []

        optimizer = Optimizer(review_logs)

        optimal_parameters = optimizer.compute_optimal_parameters()

        assert optimal_parameters == starting_parameters