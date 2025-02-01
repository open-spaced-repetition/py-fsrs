from fsrs import ReviewLog, Optimizer, DEFAULT_PARAMETERS
import pandas as pd
from copy import deepcopy
from random import shuffle
import numpy as np


def get_revlogs() -> list[ReviewLog]:
    """
    reads a csv of prepared exported anki review logs
    and returns them as a list of ReviewLog objects
    """

    df = pd.read_csv("tests/review_logs_josh_1711744352250_to_1728234780857.csv")

    review_logs = []
    for index, row in df.iterrows():
        card_id = row["card_id"]
        rating = row["review_rating"]
        review_datetime = row["review_time"]
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

        expected_optimal_parameters = [
            0.2363282014982659,
            1.18385,
            2.803907798259358,
            15.69105,
            7.450626954515589,
            0.2002981626622733,
            1.6499680903504104,
            0.030489930904182852,
            1.3726592620867732,
            0.20407416745070098,
            0.9003453768459656,
            2.0169172501722157,
            0.05052109238927132,
            0.249798385275728,
            2.3878771773930296,
            0.47499255667044843,
            2.9898,
            0.19075214884576136,
            1.0712483452116681,
        ]

        optimizer = Optimizer(review_logs)

        optimal_parameters = optimizer.compute_optimal_parameters()

        # the optimal paramaters are no longer equal to the starting parameters
        assert optimal_parameters != list(DEFAULT_PARAMETERS)

        # the output is expected
        assert np.allclose(optimal_parameters, expected_optimal_parameters)

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

        optimal_parameters = [
            0.2363282014982659,
            1.18385,
            2.803907798259358,
            15.69105,
            7.450626954515589,
            0.2002981626622733,
            1.6499680903504104,
            0.030489930904182852,
            1.3726592620867732,
            0.20407416745070098,
            0.9003453768459656,
            2.0169172501722157,
            0.05052109238927132,
            0.249798385275728,
            2.3878771773930296,
            0.47499255667044843,
            2.9898,
            0.19075214884576136,
            1.0712483452116681,
        ]

        optimal_retention_optimal_parameters = optimizer.compute_optimal_retention(
            parameters=optimal_parameters
        )

        # deterministic outcome
        assert optimal_retention_optimal_parameters == expected_optimal_retention

        # computing the optimal retention on a new optimizer with the same review logs and parameters will return
        # the same result
        optimizer_2 = Optimizer(review_logs)
        optimal_retention_optimal_parameters_2 = optimizer_2.compute_optimal_retention(
            parameters=optimal_parameters
        )
        assert (
            optimal_retention_optimal_parameters_2
            == optimal_retention_optimal_parameters
        )

        # computing the optimal retention with a different set of parameters can yield a different result
        optimal_retention_default_parameters = optimizer_2.compute_optimal_retention(
            parameters=DEFAULT_PARAMETERS
        )
        assert (
            optimal_retention_default_parameters != optimal_retention_optimal_parameters
        )
