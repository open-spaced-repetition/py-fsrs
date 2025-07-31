"""
fsrs.optimizer
---------

This module defines the optional Optimizer class.
"""

from fsrs.card import Card
from fsrs.review_log import ReviewLog, Rating
from fsrs.scheduler import (
    Scheduler,
    DEFAULT_PARAMETERS,
    LOWER_BOUNDS_PARAMETERS,
    UPPER_BOUNDS_PARAMETERS,
)

import math
from datetime import datetime, timezone
from copy import deepcopy
from random import Random
from statistics import mean

try:
    import torch
    from torch.nn import BCELoss
    from torch import optim
    import pandas as pd
    from tqdm import tqdm

    # weight clipping
    LOWER_BOUNDS_PARAMETERS_TENSORS = torch.tensor(
        LOWER_BOUNDS_PARAMETERS,
        dtype=torch.float64,
    )

    UPPER_BOUNDS_PARAMETERS_TENSORS = torch.tensor(
        UPPER_BOUNDS_PARAMETERS,
        dtype=torch.float64,
    )

    # hyper parameters
    num_epochs = 5
    mini_batch_size = 512
    learning_rate = 4e-2
    max_seq_len = (
        64  # up to the first 64 reviews of each card are used for optimization
    )

    class Optimizer:
        """
        The FSRS optimizer.

        Enables the optimization of FSRS scheduler parameters from existing review logs for more accurate interval calculations.

        Attributes:
            review_logs: A collection of previous ReviewLog objects from a user.
            _revlogs_train: The collection of review logs, sorted and formatted for optimization.
        """

        review_logs: tuple[ReviewLog, ...]
        _revlogs_train: dict

        def __init__(
            self, review_logs: tuple[ReviewLog, ...] | list[ReviewLog]
        ) -> None:
            """
            Initializes the Optimizer with a set of ReviewLogs. Also formats a copy of the review logs for optimization.

            Note that the ReviewLogs provided by the user don't need to be in order.
            """

            def _format_revlogs() -> dict:
                """
                Sorts and converts the tuple of ReviewLog objects to a dictionary format for optimizing
                """

                revlogs_train = {}
                for review_log in self.review_logs:
                    # pull data out of current ReviewLog object
                    card_id = review_log.card_id
                    rating = review_log.rating
                    review_datetime = review_log.review_datetime
                    review_duration = review_log.review_duration

                    # if the card was rated Again, it was not recalled
                    recall = 0 if rating == Rating.Again else 1

                    # as a ML problem, [x, y] = [ [review_datetime, rating, review_duration], recall ]
                    datum = [[review_datetime, rating, review_duration], recall]

                    if card_id not in revlogs_train:
                        revlogs_train[card_id] = []

                    revlogs_train[card_id].append((datum))
                    revlogs_train[card_id] = sorted(
                        revlogs_train[card_id], key=lambda x: x[0][0]
                    )  # keep reviews sorted

                # sort the dictionary in order of when each card history starts
                revlogs_train = dict(sorted(revlogs_train.items()))

                return revlogs_train

            self.review_logs = deepcopy(tuple(review_logs))

            # format the ReviewLog data for optimization
            self._revlogs_train = _format_revlogs()

        def _compute_batch_loss(self, *, parameters: list[float]) -> float:
            """
            Computes the current total loss for the entire batch of review logs.
            """

            card_ids = list(self._revlogs_train.keys())
            params = torch.tensor(parameters, dtype=torch.float64)
            loss_fn = BCELoss()
            scheduler = Scheduler(parameters=params)
            step_losses = []

            for card_id in card_ids:
                card_review_history = self._revlogs_train[card_id][:max_seq_len]

                for i in range(len(card_review_history)):
                    review = card_review_history[i]

                    x_date = review[0][0]
                    y_retrievability = review[1]
                    u_rating = review[0][1]

                    if i == 0:
                        card = Card(card_id=card_id, due=x_date)

                    y_pred_retrievability = scheduler.get_card_retrievability(
                        card=card, current_datetime=x_date
                    )
                    y_retrievability = torch.tensor(
                        y_retrievability, dtype=torch.float64
                    )

                    if card.last_review and (x_date - card.last_review).days > 0:
                        step_loss = loss_fn(y_pred_retrievability, y_retrievability)
                        step_losses.append(step_loss)

                    card, _ = scheduler.review_card(
                        card=card,
                        rating=u_rating,
                        review_datetime=x_date,
                        review_duration=None,
                    )

            batch_loss = torch.sum(torch.stack(step_losses))
            batch_loss = batch_loss.item() / len(step_losses)

            return batch_loss

        def compute_optimal_parameters(self, verbose: bool = False) -> list[float]:
            """
            Computes a set of optimized parameters for the FSRS scheduler and returns it as a list of floats.

            High level explanation of optimization:
            ---------------------------------------
            FSRS is a many-to-many sequence model where the "State" at each step is a Card object at a given point in time,
            the input is the time of the review and the output is the predicted retrievability of the card at the time of review.

            Each card's review history can be thought of as a sequence, each review as a step and each collection of card review histories
            as a batch.

            The loss is computed by comparing the predicted retrievability of the Card at each step with whether the Card was actually
            sucessfully recalled or not (0/1).

            Finally, the card objects at each step in their sequences are updated using the current parameters of the Scheduler
            as well as the rating given to that card by the user. The parameters of the Scheduler is what is being optimized.
            """

            def _num_reviews() -> int:
                """
                Computes how many Review-state reviews there are in the dataset.
                Only the loss from Review-state reviews count for optimization and their number must
                be computed in advance to properly initialize the Cosine Annealing learning rate scheduler.
                """

                scheduler = Scheduler()
                num_reviews = 0
                # iterate through the card review histories
                card_ids = list(self._revlogs_train.keys())
                for card_id in card_ids:
                    card_review_history = self._revlogs_train[card_id][:max_seq_len]

                    # iterate through the current Card's review history
                    for i in range(len(card_review_history)):
                        review = card_review_history[i]

                        review_datetime = review[0][0]
                        rating = review[0][1]

                        # if this is the first review, create the Card object
                        if i == 0:
                            card = Card(card_id=card_id, due=review_datetime)

                        # only non-same-day reviews count
                        if (
                            card.last_review
                            and (review_datetime - card.last_review).days > 0
                        ):
                            num_reviews += 1

                        card, _ = scheduler.review_card(
                            card=card,
                            rating=rating,
                            review_datetime=review_datetime,
                            review_duration=None,
                        )

                return num_reviews

            def _update_parameters(
                *, step_losses: list,
                adam_optimizer: torch.optim.Adam,
                params: torch.Tensor,
                lr_scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
            ) -> None:
                """
                Computes and updates the current FSRS parameters based on the step losses. Also updates the learning rate scheduler.
                """

                # Backpropagate through the loss
                mini_batch_loss = torch.sum(torch.stack(step_losses))
                adam_optimizer.zero_grad()  # clear previous gradients
                mini_batch_loss.backward()  # compute gradients
                adam_optimizer.step()  # Update parameters

                # clamp the weights in place without modifying the computational graph
                with torch.no_grad():
                    params.clamp_(
                        min=LOWER_BOUNDS_PARAMETERS_TENSORS,
                        max=UPPER_BOUNDS_PARAMETERS_TENSORS,
                    )

                # update the learning rate
                lr_scheduler.step()

            # set local random seed for reproducibility
            rng = Random(42)

            card_ids = list(self._revlogs_train.keys())

            num_reviews = _num_reviews()

            if num_reviews < mini_batch_size:
                return list(DEFAULT_PARAMETERS)

            # Define FSRS Scheduler parameters as torch tensors with gradients
            params = torch.tensor(
                DEFAULT_PARAMETERS, requires_grad=True, dtype=torch.float64
            )

            loss_fn = BCELoss()
            adam_optimizer = optim.Adam([params], lr=learning_rate)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=adam_optimizer,
                T_max=math.ceil(num_reviews / mini_batch_size) * num_epochs,
            )

            best_params = None
            best_loss = math.inf
            # iterate through the epochs
            for _ in tqdm(
                range(num_epochs),
                desc="Optimizing",
                unit="epoch",
                disable=(not verbose),
            ):
                # randomly shuffle the order of which Card's review histories get computed first
                # at the beginning of each new epoch
                rng.shuffle(card_ids)

                # initialize new scheduler with updated parameters each epoch
                scheduler = Scheduler(parameters=params)

                # stores the computed loss of each individual review
                step_losses = []

                # iterate through the card review histories (sequences)
                for card_id in card_ids:
                    card_review_history = self._revlogs_train[card_id][:max_seq_len]

                    # iterate through the current Card's review history (steps)
                    for i in range(len(card_review_history)):
                        review = card_review_history[i]

                        # input
                        x_date = review[0][0]
                        # target
                        y_retrievability = review[1]
                        # update
                        u_rating = review[0][1]

                        # if this is the first review, create the Card object
                        if i == 0:
                            card = Card(card_id=card_id, due=x_date)

                        # predicted target
                        y_pred_retrievability = scheduler.get_card_retrievability(
                            card=card, current_datetime=x_date
                        )
                        y_retrievability = torch.tensor(
                            y_retrievability, dtype=torch.float64
                        )

                        # only compute step-loss on non-same-day reviews
                        if card.last_review and (x_date - card.last_review).days > 0:
                            step_loss = loss_fn(y_pred_retrievability, y_retrievability)
                            step_losses.append(step_loss)

                        # update the card's state
                        card, _ = scheduler.review_card(
                            card=card,
                            rating=u_rating,
                            review_datetime=x_date,
                            review_duration=None,
                        )

                        # take a gradient step after each mini-batch
                        if len(step_losses) == mini_batch_size:
                            _update_parameters(
                                step_losses=step_losses,
                                adam_optimizer=adam_optimizer,
                                params=params,
                                lr_scheduler=lr_scheduler,
                            )

                            # update the scheduler's with the new parameters
                            scheduler = Scheduler(parameters=params)
                            # clear the step losses for next batch
                            step_losses = []

                            # remove gradient history from tensor card parameters for next batch
                            card.stability = card.stability.detach()
                            card.difficulty = card.difficulty.detach()

                # update params on remaining review logs
                if len(step_losses) > 0:
                    _update_parameters(
                        step_losses=step_losses,
                        adam_optimizer=adam_optimizer,
                        params=params,
                        lr_scheduler=lr_scheduler,
                    )

                # compute the current batch loss after each epoch
                detached_params = [
                    x.detach().item() for x in list(params.detach())
                ]  # convert to floats
                with torch.no_grad():
                    epoch_batch_loss = self._compute_batch_loss(
                        parameters=detached_params
                    )

                # if the batch loss is better with the current parameters, update the current best parameters
                if epoch_batch_loss < best_loss:
                    best_loss = epoch_batch_loss
                    best_params = detached_params

            return best_params

        def _compute_probs_and_costs(self) -> dict[str, float]:
            review_log_df = pd.DataFrame(
                vars(review_log) for review_log in self.review_logs
            )

            review_log_df = review_log_df.sort_values(
                by=["card_id", "review_datetime"], ascending=[True, True]
            ).reset_index(drop=True)

            # dictionary to return
            probs_and_costs_dict = {}

            # compute the probabilities and costs of the first rating
            first_reviews_df = review_log_df.loc[
                ~review_log_df["card_id"].duplicated(keep="first")
            ].reset_index(drop=True)

            first_again_reviews_df = first_reviews_df.loc[
                first_reviews_df["rating"] == Rating.Again
            ]
            first_hard_reviews_df = first_reviews_df.loc[
                first_reviews_df["rating"] == Rating.Hard
            ]
            first_good_reviews_df = first_reviews_df.loc[
                first_reviews_df["rating"] == Rating.Good
            ]
            first_easy_reviews_df = first_reviews_df.loc[
                first_reviews_df["rating"] == Rating.Easy
            ]

            # compute the probability of the user clicking again/hard/good/easy given it's their first review
            num_first_again = len(first_again_reviews_df)
            num_first_hard = len(first_hard_reviews_df)
            num_first_good = len(first_good_reviews_df)
            num_first_easy = len(first_easy_reviews_df)

            num_first_review = (
                num_first_again + num_first_hard + num_first_good + num_first_easy
            )

            prob_first_again = num_first_again / num_first_review
            prob_first_hard = num_first_hard / num_first_review
            prob_first_good = num_first_good / num_first_review
            prob_first_easy = num_first_easy / num_first_review

            probs_and_costs_dict["prob_first_again"] = prob_first_again
            probs_and_costs_dict["prob_first_hard"] = prob_first_hard
            probs_and_costs_dict["prob_first_good"] = prob_first_good
            probs_and_costs_dict["prob_first_easy"] = prob_first_easy

            # compute the cost of the user clicking again/hard/good/easy on their first review
            first_again_review_durations = list(
                first_again_reviews_df["review_duration"]
            )
            first_hard_review_durations = list(first_hard_reviews_df["review_duration"])
            first_good_review_durations = list(first_good_reviews_df["review_duration"])
            first_easy_review_durations = list(first_easy_reviews_df["review_duration"])

            avg_first_again_review_duration = (
                mean(first_again_review_durations)
                if first_again_review_durations
                else 0
            )
            avg_first_hard_review_duration = (
                mean(first_hard_review_durations) if first_hard_review_durations else 0
            )
            avg_first_good_review_duration = (
                mean(first_good_review_durations) if first_good_review_durations else 0
            )
            avg_first_easy_review_duration = (
                mean(first_easy_review_durations) if first_easy_review_durations else 0
            )

            probs_and_costs_dict["avg_first_again_review_duration"] = (
                avg_first_again_review_duration
            )
            probs_and_costs_dict["avg_first_hard_review_duration"] = (
                avg_first_hard_review_duration
            )
            probs_and_costs_dict["avg_first_good_review_duration"] = (
                avg_first_good_review_duration
            )
            probs_and_costs_dict["avg_first_easy_review_duration"] = (
                avg_first_easy_review_duration
            )

            # compute the probabilities and costs of non-first ratings
            non_first_reviews_df = review_log_df.loc[
                review_log_df["card_id"].duplicated(keep="first")
            ].reset_index(drop=True)

            again_reviews_df = non_first_reviews_df.loc[
                non_first_reviews_df["rating"] == Rating.Again
            ]
            hard_reviews_df = non_first_reviews_df.loc[
                non_first_reviews_df["rating"] == Rating.Hard
            ]
            good_reviews_df = non_first_reviews_df.loc[
                non_first_reviews_df["rating"] == Rating.Good
            ]
            easy_reviews_df = non_first_reviews_df.loc[
                non_first_reviews_df["rating"] == Rating.Easy
            ]

            # compute the probability of the user clicking hard/good/easy given they correctly recalled the card
            num_hard = len(hard_reviews_df)
            num_good = len(good_reviews_df)
            num_easy = len(easy_reviews_df)

            num_recall = num_hard + num_good + num_easy

            prob_hard = num_hard / num_recall
            prob_good = num_good / num_recall
            prob_easy = num_easy / num_recall

            probs_and_costs_dict["prob_hard"] = prob_hard
            probs_and_costs_dict["prob_good"] = prob_good
            probs_and_costs_dict["prob_easy"] = prob_easy

            again_review_durations = list(again_reviews_df["review_duration"])
            hard_review_durations = list(hard_reviews_df["review_duration"])
            good_review_durations = list(good_reviews_df["review_duration"])
            easy_review_durations = list(easy_reviews_df["review_duration"])

            avg_again_review_duration = (
                mean(again_review_durations) if again_review_durations else 0
            )
            avg_hard_review_duration = (
                mean(hard_review_durations) if hard_review_durations else 0
            )
            avg_good_review_duration = (
                mean(good_review_durations) if good_review_durations else 0
            )
            avg_easy_review_duration = (
                mean(easy_review_durations) if easy_review_durations else 0
            )

            probs_and_costs_dict["avg_again_review_duration"] = (
                avg_again_review_duration
            )
            probs_and_costs_dict["avg_hard_review_duration"] = avg_hard_review_duration
            probs_and_costs_dict["avg_good_review_duration"] = avg_good_review_duration
            probs_and_costs_dict["avg_easy_review_duration"] = avg_easy_review_duration

            return probs_and_costs_dict

        def _simulate_cost(
            self,
            *, desired_retention: float,
            parameters: tuple[float, ...] | list[float],
            num_cards_simulate: int,
            probs_and_costs_dict: dict[str, float],
        ) -> float:
            rng = Random(42)

            # simulate from the beginning of 2025 till before the beginning of 2026
            start_date = datetime(2025, 1, 1, 0, 0, 0, 0, timezone.utc)
            end_date = datetime(2026, 1, 1, 0, 0, 0, 0, timezone.utc)

            scheduler = Scheduler(
                parameters=parameters,
                desired_retention=desired_retention,
                enable_fuzzing=False,
            )

            # unpack probs_and_costs_dict
            prob_first_again = probs_and_costs_dict["prob_first_again"]
            prob_first_hard = probs_and_costs_dict["prob_first_hard"]
            prob_first_good = probs_and_costs_dict["prob_first_good"]
            prob_first_easy = probs_and_costs_dict["prob_first_easy"]

            avg_first_again_review_duration = probs_and_costs_dict[
                "avg_first_again_review_duration"
            ]
            avg_first_hard_review_duration = probs_and_costs_dict[
                "avg_first_hard_review_duration"
            ]
            avg_first_good_review_duration = probs_and_costs_dict[
                "avg_first_good_review_duration"
            ]
            avg_first_easy_review_duration = probs_and_costs_dict[
                "avg_first_easy_review_duration"
            ]

            prob_hard = probs_and_costs_dict["prob_hard"]
            prob_good = probs_and_costs_dict["prob_good"]
            prob_easy = probs_and_costs_dict["prob_easy"]

            avg_again_review_duration = probs_and_costs_dict[
                "avg_again_review_duration"
            ]
            avg_hard_review_duration = probs_and_costs_dict["avg_hard_review_duration"]
            avg_good_review_duration = probs_and_costs_dict["avg_good_review_duration"]
            avg_easy_review_duration = probs_and_costs_dict["avg_easy_review_duration"]

            simulation_cost = 0
            for i in range(num_cards_simulate):
                card = Card()
                curr_date = start_date
                while curr_date < end_date:
                    # the card is new
                    if curr_date == start_date:
                        rating = rng.choices(
                            [Rating.Again, Rating.Hard, Rating.Good, Rating.Easy],
                            weights=[
                                prob_first_again,
                                prob_first_hard,
                                prob_first_good,
                                prob_first_easy,
                            ],
                        )[0]

                        if rating == Rating.Again:
                            simulation_cost += avg_first_again_review_duration

                        elif rating == Rating.Hard:
                            simulation_cost += avg_first_hard_review_duration

                        elif rating == Rating.Good:
                            simulation_cost += avg_first_good_review_duration

                        elif rating == Rating.Easy:
                            simulation_cost += avg_first_easy_review_duration

                    # the card is not new
                    else:
                        rating = rng.choices(
                            ["recall", Rating.Again],
                            weights=[desired_retention, 1.0 - desired_retention],
                        )[0]

                        if rating == "recall":
                            # compute probability that the user chose hard/good/easy, GIVEN that they correctly recalled the card
                            rating = rng.choices(
                                [Rating.Hard, Rating.Good, Rating.Easy],
                                weights=[prob_hard, prob_good, prob_easy],
                            )[0]

                        if rating == Rating.Again:
                            simulation_cost += avg_again_review_duration

                        elif rating == Rating.Hard:
                            simulation_cost += avg_hard_review_duration

                        elif rating == Rating.Good:
                            simulation_cost += avg_good_review_duration

                        elif rating == Rating.Easy:
                            simulation_cost += avg_easy_review_duration

                    card, _ = scheduler.review_card(
                        card=card, rating=rating, review_datetime=curr_date
                    )
                    curr_date = card.due

            total_knowledge = desired_retention * num_cards_simulate
            simulation_cost = simulation_cost / total_knowledge

            return simulation_cost

        def compute_optimal_retention(
            self, parameters: tuple[float, ...] | list[float]
        ) -> list[float]:
            def _validate_review_logs() -> None:
                if len(self.review_logs) < 512:
                    raise ValueError(
                        "Not enough ReviewLog's: at least 512 ReviewLog objects are required to compute optimal retention"
                    )

                for review_log in self.review_logs:
                    if review_log.review_duration is None:
                        raise ValueError(
                            "ReviewLog.review_duration cannot be None when computing optimal retention"
                        )

            _validate_review_logs()

            NUM_CARDS_SIMULATE = 1000
            DESIRED_RETENTIONS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

            probs_and_costs_dict = self._compute_probs_and_costs()

            simulation_costs = []
            for desired_retention in DESIRED_RETENTIONS:
                simulation_cost = self._simulate_cost(
                    desired_retention=desired_retention,
                    parameters=parameters,
                    num_cards_simulate=NUM_CARDS_SIMULATE,
                    probs_and_costs_dict=probs_and_costs_dict,
                )
                simulation_costs.append(simulation_cost)

            min_index = simulation_costs.index(min(simulation_costs))
            optimal_retention = DESIRED_RETENTIONS[min_index]

            return optimal_retention

except ImportError:

    class Optimizer:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                'Optimizer is not installed.\nInstall it with: pip install "fsrs[optimizer]"'
            )


__all__ = ["Optimizer"]
