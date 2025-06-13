from fsrs.optimizer import Optimizer
from fsrs.review_log import ReviewLog, Rating
from fsrs.scheduler import Scheduler, DEFAULT_PARAMETERS
from fsrs.card import Card

import pandas as pd
from copy import deepcopy
from random import shuffle
import numpy as np
import pytest
from datetime import datetime, timezone, timedelta


test_optimal_parameters = [
    0.07181179205417863,
    1.1771,
    3.016031528328984,
    16.1507,
    7.313588988324146,
    0.3119116156472349,
    2.1325475249616583,
    0.02668484845216306,
    1.3685438167248498,
    0.03286122728404283,
    0.8861484775736176,
    1.856174600815682,
    0.09059541606248055,
    0.27523877674548947,
    2.3456405140066834,
    0.4568750726208301,
    3.0004,
    0.7787402478565422,
    0.31745415553393075,
    0.24515670639930176,
    0.5849145294103553,
]


def get_revlogs() -> list[ReviewLog]:
    """
    reads a csv of prepared exported anki review logs
    and returns them as a list of ReviewLog objects
    """

    df = pd.read_csv("tests/review_logs_josh_1711744352250_to_1728234780857.csv")

    review_logs = []
    for index, row in df.iterrows():
        card_id = row["card_id"]
        rating = Rating(row["review_rating"])
        review_datetime = datetime.fromisoformat(row["review_time"])
        review_duration = row["review_duration"]

        review_log = ReviewLog(
            card_id=card_id,
            rating=rating,
            review_datetime=review_datetime,
            review_duration=review_duration,
        )

        review_logs.append(review_log)

    return review_logs


class TestOptimizer:
    def test_zero_revlogs(self):
        """
        if no review logs are provided, the optimal parameters should not change
        from the starting parameters
        """

        review_logs = []

        optimizer = Optimizer(review_logs)

        optimal_parameters = optimizer.compute_optimal_parameters()

        assert optimal_parameters == list(DEFAULT_PARAMETERS)

    def test_review_logs(self):
        """
        test the optimizer on a set of exported anki review logs
        """

        review_logs = get_revlogs()

        optimizer = Optimizer(review_logs)

        optimal_parameters = optimizer.compute_optimal_parameters()

        # the optimal paramaters are no longer equal to the starting parameters
        assert optimal_parameters != list(DEFAULT_PARAMETERS)

        # the output is expected and deterministic
        assert np.allclose(optimal_parameters, test_optimal_parameters)

        # the computed loss with the optimized parameters are less than that of the starting parameters
        starting_loss = optimizer._compute_batch_loss(DEFAULT_PARAMETERS)
        optimized_loss = optimizer._compute_batch_loss(optimal_parameters)
        assert optimized_loss < starting_loss

        # calling the same optimizer again will yield the same parameters
        optimal_parameters_again = optimizer.compute_optimal_parameters()
        assert optimal_parameters == optimal_parameters_again

        # initializing another optimizer will give the same results as the original optimizer
        optimizer_new = Optimizer(review_logs)
        optimal_parameters_new = optimizer_new.compute_optimal_parameters()
        assert optimal_parameters == optimal_parameters_new

    def test_few_review_logs(self):
        """
        if very few review logs are provided to the optimizer, return unchanged default parameters
        """

        review_logs = get_revlogs()

        # if there are fewer review logs than the minibatch size of 512, then the parameters returned are the starting parameters
        few_revlogs = review_logs[0:500]

        optimizer = Optimizer(few_revlogs)

        optimal_parameters = optimizer.compute_optimal_parameters()

        assert optimal_parameters == list(DEFAULT_PARAMETERS)

    def test_unordered_review_logs(self):
        """
        the order of the ReviewLog objects passed to the Optimizer doesn't matter
        as they're sorted within the object
        """

        review_logs = get_revlogs()

        reviewlogs_copy1 = deepcopy(review_logs)
        reviewlogs_copy2 = deepcopy(review_logs)

        # randomly shuffle the review logs
        shuffle(reviewlogs_copy1)
        shuffle(reviewlogs_copy2)

        optimizer1 = Optimizer(reviewlogs_copy1)
        optimizer2 = Optimizer(reviewlogs_copy2)

        optimal_parameters1 = optimizer1.compute_optimal_parameters()
        optimal_parameters2 = optimizer2.compute_optimal_parameters()

        assert optimal_parameters1 == optimal_parameters2

    def test_optimal_retention(self):
        review_logs = get_revlogs()

        optimizer = Optimizer(review_logs)

        expected_optimal_retention = 0.85

        optimal_retention_optimal_parameters = optimizer.compute_optimal_retention(
            parameters=test_optimal_parameters
        )

        # deterministic outcome
        assert optimal_retention_optimal_parameters == expected_optimal_retention

        # computing the optimal retention on a new optimizer with the same review logs and parameters will return
        # the same result
        optimizer_2 = Optimizer(review_logs)
        optimal_retention_optimal_parameters_2 = optimizer_2.compute_optimal_retention(
            parameters=test_optimal_parameters
        )
        assert (
            optimal_retention_optimal_parameters_2
            == optimal_retention_optimal_parameters
        )

        # computing the optimal retention with a different set of parameters can yield a different result
        parameters_2 = [
            0.01,
            0.01,
            0.01,
            0.01,
            1.0,
            0.1,
            0.1,
            0.01,
            0.0,
            0.0,
            0.01,
            0.1,
            0.01,
            0.01,
            0.01,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.1,
        ]
        optimal_retention_parameters_2 = optimizer_2.compute_optimal_retention(
            parameters=parameters_2
        )
        assert optimal_retention_parameters_2 != optimal_retention_optimal_parameters

    def test_optimal_retention_zero_review_logs(self):
        # can't compute optimal retention with zero review logs
        zero_revlogs = []
        optimizer = Optimizer(zero_revlogs)
        with pytest.raises(ValueError):
            _ = optimizer.compute_optimal_retention(parameters=DEFAULT_PARAMETERS)

    def test_optimal_retention_few_review_logs(self):
        review_logs = get_revlogs()
        few_revlogs = review_logs[:100]

        optimizer = Optimizer(few_revlogs)
        with pytest.raises(ValueError):
            _ = optimizer.compute_optimal_retention(parameters=DEFAULT_PARAMETERS)

    def test_optimal_retention_no_review_duration(self):
        review_logs = get_revlogs()

        review_log_without_review_duration = ReviewLog(
            card_id=42,
            rating=2,
            review_datetime=datetime(2025, 1, 1, 0, 0, 0, 0, timezone.utc),
            review_duration=None,
        )

        review_logs.append(review_log_without_review_duration)

        optimizer = Optimizer(review_logs)
        with pytest.raises(ValueError):
            _ = optimizer.compute_optimal_retention(parameters=DEFAULT_PARAMETERS)

    def test_simulated_costs(self):
        review_logs = get_revlogs()

        optimizer = Optimizer(review_logs)

        probs_and_costs_dict = optimizer._compute_probs_and_costs()

        simulation_cost_0_75 = optimizer._simulate_cost(
            desired_retention=0.75,
            parameters=test_optimal_parameters,
            num_cards_simulate=1000,
            probs_and_costs_dict=probs_and_costs_dict,
        )

        assert round(simulation_cost_0_75) == 250026

        simulation_cost_0_85 = optimizer._simulate_cost(
            desired_retention=0.85,
            parameters=test_optimal_parameters,
            num_cards_simulate=1000,
            probs_and_costs_dict=probs_and_costs_dict,
        )

        assert round(simulation_cost_0_85) == 208594

        simulation_cost_0_95 = optimizer._simulate_cost(
            desired_retention=0.95,
            parameters=test_optimal_parameters,
            num_cards_simulate=1000,
            probs_and_costs_dict=probs_and_costs_dict,
        )

        assert round(simulation_cost_0_95) == 255857

        # holds true for these specific revlogs
        assert simulation_cost_0_85 <= simulation_cost_0_75
        assert simulation_cost_0_85 <= simulation_cost_0_95

    def test_optimize_review_logs_with_difficulty_1_cards(self):
        """
        Create hypothetical review logs where cards have
        their difficulty values driven down to 1.0 after repeated easy ratings.
        """

        scheduler = Scheduler()
        review_logs = []
        for _ in range(100):
            card = Card()
            for day in range(100):
                card, review_log = scheduler.review_card(
                    card=card,
                    rating=Rating.Easy,
                    review_datetime=datetime(2022, 11, 29, 12, 30, 0, 0, timezone.utc)
                    + timedelta(days=day),
                )
                review_logs.append(review_log)

            assert card.difficulty == 1.0

        optimizer = Optimizer(review_logs)
        _ = optimizer.compute_optimal_parameters()  # this should not raise an exception
