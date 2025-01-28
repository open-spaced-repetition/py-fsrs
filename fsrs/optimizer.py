"""
fsrs.optimizer
---------

This module defines the optional Optimizer class.
"""

from .fsrs import Card, ReviewLog, Scheduler, Rating, DEFAULT_PARAMETERS
import math
from datetime import datetime
from copy import deepcopy
from random import Random

try:
    import torch
    from torch.nn import BCELoss
    from torch import optim

    # weight clipping
    S_MIN = 0.01
    INIT_S_MAX = 100.0
    lower_bounds = torch.tensor(
        [
            S_MIN,
            S_MIN,
            S_MIN,
            S_MIN,
            1.0,
            0.1,
            0.1,
            0.0,
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
        ],
        dtype=torch.float64,
    )
    upper_bounds = torch.tensor(
        [
            INIT_S_MAX,
            INIT_S_MAX,
            INIT_S_MAX,
            INIT_S_MAX,
            10.0,
            4.0,
            4.0,
            0.75,
            4.5,
            0.8,
            3.5,
            5.0,
            0.25,
            0.9,
            4.0,
            1.0,
            6.0,
            2.0,
            2.0,
        ],
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
            review_logs (tuple[ReviewLog, ...]): A collection of previous ReviewLog objects from a user.
            _revlogs_train (dict): The collection of review logs, sorted and formatted for optimization.
        """

        review_logs: tuple[ReviewLog, ...]
        _revlogs_train: dict

        def __init__(
            self, review_logs: tuple[ReviewLog, ...] | list[ReviewLog]
        ) -> None:
            """
            Initializes the Optimizer with a set of ReviewLogs. Also formats an copy of the review logs for optimization.

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

                # convert the timestamps in the json from isoformat to datetime variables
                for key, values in revlogs_train.items():
                    for entry in values:
                        entry[0][0] = datetime.fromisoformat(entry[0][0])

                # sort the dictionary in order of when each card history starts
                revlogs_train = dict(sorted(revlogs_train.items()))

                return revlogs_train

            self.review_logs = deepcopy(tuple(review_logs))

            # format the ReviewLog data for optimization
            self._revlogs_train = _format_revlogs()

        def _compute_batch_loss(self, parameters: list[float]) -> float:
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
                        card = Card(due=x_date)

                    y_pred_retrievability = card.get_retrievability(x_date)
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

        def compute_optimal_parameters(self) -> list[float]:
            """
            Computes a set of 19 optimized parameters for the FSRS scheduler and returns it as a list of floats.

            High level explanation of optimization:
            ---------------------------------------
            FSRS is a many-to-many sequence model where the "State" at each step is a Card object at a given point in time,
            the input is the time of the review and the output is the predicted retrievability of the card at the time of review.

            Each card's review history can be thought of as a sequence, each review as a step and each collection of card review histories
            as a batch.

            The loss is computed by comparing the predicted retrievability of the Card at each step with whether the Card was actually
            sucessfully recalled or not (0/1).

            Finally, the card objects at each step in their sequences are updated using the 19 current parameters of the Scheduler
            as well as the rating given to that card by the user. The 19 parameters of the Scheduler is what is being optimized.
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
                            card = Card(due=review_datetime)

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
                step_losses: list,
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
                    params.clamp_(min=lower_bounds, max=upper_bounds)

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
            for j in range(num_epochs):
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
                            card = Card(due=x_date)

                        # predicted target
                        y_pred_retrievability = card.get_retrievability(x_date)
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

except ImportError:

    class Optimizer:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("The Optimizer class requires torch be installed.")
