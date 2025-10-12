from fsrs.scheduler import Scheduler, STABILITY_MIN, DEFAULT_PARAMETERS
from fsrs.card import Card, State
from fsrs.review_log import ReviewLog, Rating

from datetime import datetime, timedelta, timezone
import json
import pytest
import random
from copy import deepcopy
import sys
import re

TEST_RATINGS_1 = (
    Rating.Good,
    Rating.Good,
    Rating.Good,
    Rating.Good,
    Rating.Good,
    Rating.Good,
    Rating.Again,
    Rating.Again,
    Rating.Good,
    Rating.Good,
    Rating.Good,
    Rating.Good,
    Rating.Good,
)


class TestPyFSRS:
    def test_review_card(self):
        scheduler = Scheduler(enable_fuzzing=False)

        card = Card()
        review_datetime = datetime(2022, 11, 29, 12, 30, 0, 0, timezone.utc)

        ivl_history = []
        for rating in TEST_RATINGS_1:
            card, _ = scheduler.review_card(
                card=card, rating=rating, review_datetime=review_datetime
            )

            ivl = (card.due - card.last_review).days
            ivl_history.append(ivl)

            review_datetime = card.due

        assert ivl_history == [0, 2, 11, 46, 163, 498, 0, 0, 2, 4, 7, 12, 21]

    def test_repeated_correct_reviews(self):
        scheduler = Scheduler(enable_fuzzing=False)

        card = Card()
        review_datetimes = [
            datetime(2022, 11, 29, 12, 30, 0, i, timezone.utc) for i in range(10)
        ]

        for review_datetime in review_datetimes:
            card, _ = scheduler.review_card(
                card=card, rating=Rating.Easy, review_datetime=review_datetime
            )

        assert card.difficulty == 1.0

    def test_memo_state(self):
        scheduler = Scheduler()

        ratings = (
            Rating.Again,
            Rating.Good,
            Rating.Good,
            Rating.Good,
            Rating.Good,
            Rating.Good,
        )
        ivl_history = [0, 0, 1, 3, 8, 21]

        card = Card()
        review_datetime = datetime(2022, 11, 29, 12, 30, 0, 0, timezone.utc)

        for rating, ivl in zip(ratings, ivl_history):
            review_datetime += timedelta(days=ivl)
            card, _ = scheduler.review_card(
                card=card, rating=rating, review_datetime=review_datetime
            )

        assert card.stability == pytest.approx(53.62691, abs=1e-4)
        assert card.difficulty == pytest.approx(6.3574867, abs=1e-4)

    def test_repeat_default_arg(self):
        scheduler = Scheduler()

        card = Card()

        rating = Rating.Good

        card, _ = scheduler.review_card(
            card=card,
            rating=rating,
        )

        due = card.due

        time_delta = due - datetime.now(timezone.utc)

        assert time_delta.seconds > 500  # due in approx. 8-10 minutes

    def test_datetime(self):
        scheduler = Scheduler()
        card = Card()

        # new cards should be due immediately after creation
        assert datetime.now(timezone.utc) >= card.due

        # comparing timezone aware cards with deprecated datetime.utcnow() should raise a TypeError
        with pytest.raises(TypeError):
            datetime.now() >= card.due

        # repeating a card with a non-utc, non-timezone-aware datetime object should raise a Value Error
        with pytest.raises(ValueError):
            scheduler.review_card(
                card=card,
                rating=Rating.Good,
                review_datetime=datetime(2022, 11, 29, 12, 30, 0, 0),
            )

        # review a card with rating good before next tests
        card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=datetime.now(timezone.utc)
        )

        # card object's due and last_review attributes must be timezone aware and UTC
        assert card.due.tzinfo == timezone.utc
        assert card.last_review.tzinfo == timezone.utc
        # card object's due datetime should be later than its last review
        assert card.due >= card.last_review

    def test_Card_dict_serialize(self):
        scheduler = Scheduler()

        # create card object the normal way
        card = Card()

        # card object is not naturally JSON serializable
        with pytest.raises(TypeError):
            json.dumps(card.__dict__)

        # card object's to_dict() method makes it JSON serializable
        assert type(json.dumps(card.to_dict())) is str

        # we can reconstruct a copy of the card object equivalent to the original
        card_dict = card.to_dict()
        copied_card = Card.from_dict(card_dict)

        assert vars(card) == vars(copied_card)
        assert card.to_dict() == copied_card.to_dict()

        # (x2) perform the above tests once more with a repeated card
        reviewed_card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=datetime.now(timezone.utc)
        )

        with pytest.raises(TypeError):
            json.dumps(reviewed_card.__dict__)

        assert type(json.dumps(reviewed_card.to_dict())) is str

        reviewed_card_dict = reviewed_card.to_dict()
        copied_reviewed_card = Card.from_dict(reviewed_card_dict)

        assert vars(reviewed_card) == vars(copied_reviewed_card)
        assert reviewed_card.to_dict() == copied_reviewed_card.to_dict()

        # original card and repeated card are different
        assert vars(card) != vars(reviewed_card)
        assert card.to_dict() != reviewed_card.to_dict()

    def test_Card_json_serialize(self):
        scheduler = Scheduler()

        # create card object the normal way
        card = Card()

        # we can reconstruct a copy of the card object equivalent to the original
        card_json = card.to_json()
        copied_card = Card.from_json(card_json)

        assert card == copied_card
        assert card.to_json() == copied_card.to_json()

        # (x2) perform the above tests once more with a repeated card
        reviewed_card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=datetime.now(timezone.utc)
        )

        reviewed_card_json = reviewed_card.to_json()
        copied_reviewed_card = Card.from_json(reviewed_card_json)

        assert reviewed_card == copied_reviewed_card
        assert reviewed_card.to_json() == copied_reviewed_card.to_json()

        # original card and repeated card are different
        assert card != reviewed_card
        assert card.to_json() != reviewed_card.to_json()

    def test_ReviewLog_dict_serialize(self):
        scheduler = Scheduler()

        card = Card()

        # review a card to get the review_log
        card, review_log = scheduler.review_card(card=card, rating=Rating.Again)

        # ReviewLog object is not naturally JSON serializable
        with pytest.raises(TypeError):
            json.dumps(review_log.__dict__)

        # review_log object's to_dict() method makes it JSON serializable
        assert type(json.dumps(review_log.to_dict())) is str

        # we can reconstruct a copy of the review_log object equivalent to the original
        review_log_dict = review_log.to_dict()
        copied_review_log = ReviewLog.from_dict(review_log_dict)
        assert review_log.to_dict() == copied_review_log.to_dict()

        # (x2) perform the above tests once more with a review_log from a reviewed card
        rating = Rating.Good
        card, next_review_log = scheduler.review_card(
            card=card, rating=rating, review_datetime=datetime.now(timezone.utc)
        )

        with pytest.raises(TypeError):
            json.dumps(next_review_log.__dict__)

        assert type(json.dumps(next_review_log.to_dict())) is str

        next_review_log_dict = next_review_log.to_dict()
        copied_next_review_log = ReviewLog.from_dict(next_review_log_dict)

        assert next_review_log.to_dict() == copied_next_review_log.to_dict()

        # original review log and next review log are different
        assert review_log.to_dict() != next_review_log.to_dict()

    def test_ReviewLog_json_serialize(self):
        scheduler = Scheduler()

        card = Card()

        card, review_log = scheduler.review_card(card=card, rating=Rating.Again)

        # ReviewLog object is not naturally JSON-serializable
        with pytest.raises(TypeError):
            json.dumps(review_log.__dict__)

        # we can reconstruct a copy of the ReviewLog object equivalent to the original
        review_log_json = review_log.to_json()
        copied_review_log = ReviewLog.from_json(review_log_json)

        assert review_log == copied_review_log
        assert review_log.to_json() == copied_review_log.to_json()

        # (x2) perform the above tests one more with a review_log from a reviewed card
        rating = Rating.Good
        _, next_review_log = scheduler.review_card(
            card=card, rating=rating, review_datetime=datetime.now(timezone.utc)
        )

        next_review_log_json = next_review_log.to_json()
        copied_next_review_log = ReviewLog.from_json(next_review_log_json)

        assert next_review_log == copied_next_review_log
        assert next_review_log.to_json() == copied_next_review_log.to_json()

        # original review log and next review lot are different
        assert review_log != next_review_log
        assert review_log.to_json() != next_review_log.to_json()

    def test_Scheduler_dict_serialize(self):
        scheduler = Scheduler()

        # Scheduler objects are json-serializable through its .to_dict() method
        assert type(json.dumps(scheduler.to_dict())) is str

        # scheduler can be serialized and de-serialized while remaining the same
        scheduler_dict = scheduler.to_dict()
        copied_scheduler = Scheduler.from_dict(scheduler_dict)
        assert vars(scheduler) == vars(copied_scheduler)
        assert scheduler.to_dict() == copied_scheduler.to_dict()

    def test_Scheduler_json_serialize(self):
        scheduler = Scheduler()

        # Scheduler objects are json-serializable through its .to_json() method
        assert type(json.dumps(scheduler.to_json())) is str

        # scheduler can be serialized and de-serialized while remaining the same
        scheduler_json = scheduler.to_json()
        copied_scheduler = Scheduler.from_json(source_json=scheduler_json)
        assert scheduler == copied_scheduler
        assert scheduler.to_json() == copied_scheduler.to_json()

    def test_custom_scheduler_args(self):
        scheduler = Scheduler(
            desired_retention=0.9,
            maximum_interval=36500,
            enable_fuzzing=False,
        )
        card = Card()
        now = datetime(2022, 11, 29, 12, 30, 0, 0, timezone.utc)

        ivl_history = []

        for rating in TEST_RATINGS_1:
            card, _ = scheduler.review_card(card, rating, now)
            ivl = (card.due - card.last_review).days
            ivl_history.append(ivl)
            now = card.due

        assert ivl_history == [0, 2, 11, 46, 163, 498, 0, 0, 2, 4, 7, 12, 21]

        # initialize another scheduler and verify parameters are properly set
        parameters2 = (
            0.1456,
            0.4186,
            1.1104,
            4.1315,
            5.2417,
            1.3098,
            0.8975,
            0.0010,
            1.5674,
            0.0567,
            0.9661,
            2.0275,
            0.1592,
            0.2446,
            1.5071,
            0.2272,
            2.8755,
            1.234,
            0.56789,
            0.1437,
            0.2,
        )
        desired_retention2 = 0.85
        maximum_interval2 = 3650
        scheduler2 = Scheduler(
            parameters=parameters2,
            desired_retention=desired_retention2,
            maximum_interval=maximum_interval2,
        )

        assert scheduler2.parameters == parameters2
        assert scheduler2.desired_retention == desired_retention2
        assert scheduler2.maximum_interval == maximum_interval2

    def test_retrievability(self):
        scheduler = Scheduler()

        card = Card()

        # retrievabiliy of New card
        assert card.state == State.Learning
        retrievability = scheduler.get_card_retrievability(card=card)
        assert retrievability == 0

        # retrievabiliy of Learning card
        card, _ = scheduler.review_card(card, Rating.Good)
        assert card.state == State.Learning
        retrievability = scheduler.get_card_retrievability(card=card)
        assert 0 <= retrievability <= 1

        # retrievabiliy of Review card
        card, _ = scheduler.review_card(card, Rating.Good)
        assert card.state == State.Review
        retrievability = scheduler.get_card_retrievability(card=card)
        assert 0 <= retrievability <= 1

        # retrievabiliy of Relearning card
        card, _ = scheduler.review_card(card, Rating.Again)
        assert card.state == State.Relearning
        retrievability = scheduler.get_card_retrievability(card=card)
        assert 0 <= retrievability <= 1

    def test_good_learning_steps(self):
        scheduler = Scheduler()

        created_at = datetime.now(timezone.utc)
        card = Card()

        assert card.state == State.Learning
        assert card.step == 0

        rating = Rating.Good
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        assert card.state == State.Learning
        assert card.step == 1
        assert (
            round((card.due - created_at).total_seconds() / 100) == 6
        )  # card is due in approx. 10 minutes (600 seconds)

        rating = Rating.Good
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )
        assert card.state == State.Review
        assert card.step is None
        assert (
            round((card.due - created_at).total_seconds() / 3600) >= 24
        )  # card is due in over a day

    def test_again_learning_steps(self):
        scheduler = Scheduler()

        created_at = datetime.now(timezone.utc)
        card = Card()

        assert card.state == State.Learning
        assert card.step == 0

        rating = Rating.Again
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        assert card.state == State.Learning
        assert card.step == 0
        assert (
            round((card.due - created_at).total_seconds() / 10) == 6
        )  # card is due in approx. 1 minute (60 seconds)

    def test_hard_learning_steps(self):
        scheduler = Scheduler()

        created_at = datetime.now(timezone.utc)
        card = Card()

        assert card.state == State.Learning
        assert card.step == 0

        rating = Rating.Hard
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        assert card.state == State.Learning
        assert card.step == 0
        assert (
            round((card.due - created_at).total_seconds() / 10) == 33
        )  # card is due in approx. 5.5 minutes (330 seconds)

    def test_easy_learning_steps(self):
        scheduler = Scheduler()

        created_at = datetime.now(timezone.utc)
        card = Card()

        assert card.state == State.Learning
        assert card.step == 0

        rating = Rating.Easy
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        assert card.state == State.Review
        assert card.step is None
        assert (
            round((card.due - created_at).total_seconds() / 86400) >= 1
        )  # card is due in at least 1 full day

    def test_review_state(self):
        scheduler = Scheduler(enable_fuzzing=False)

        card = Card()

        rating = Rating.Good
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        rating = Rating.Good
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        assert card.state == State.Review
        assert card.step is None

        prev_due = card.due
        rating = Rating.Good
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        assert card.state == State.Review
        assert (
            round((card.due - prev_due).total_seconds() / 3600) >= 24
        )  # card is due in at least 1 full day

        # rate the card again
        prev_due = card.due
        rating = Rating.Again
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        assert card.state == State.Relearning
        assert (
            round((card.due - prev_due).total_seconds() / 60) == 10
        )  # card is due in 10 minutes

    def test_relearning(self):
        scheduler = Scheduler(enable_fuzzing=False)

        card = Card()

        rating = Rating.Good
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        rating = Rating.Good
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        prev_due = card.due
        rating = Rating.Good
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        # rate the card again
        prev_due = card.due
        rating = Rating.Again
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        assert card.state == State.Relearning
        assert card.step == 0
        assert (
            round((card.due - prev_due).total_seconds() / 60) == 10
        )  # card is due in 10 minutes

        prev_due = card.due
        rating = Rating.Again
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        assert card.state == State.Relearning
        assert card.step == 0
        assert (
            round((card.due - prev_due).total_seconds() / 60) == 10
        )  # card is due in 10 minutes

        prev_due = card.due
        rating = Rating.Good
        card, _ = scheduler.review_card(
            card=card, rating=rating, review_datetime=card.due
        )

        assert card.state == State.Review
        assert card.step is None
        assert (
            round((card.due - prev_due).total_seconds() / 3600) >= 24
        )  # card is due in at least 1 full day

    def test_fuzz(self):
        scheduler = Scheduler()

        # seed 1
        random.seed(42)

        card = Card()
        card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=datetime.now(timezone.utc)
        )
        card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=card.due
        )
        prev_due = card.due
        card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=card.due
        )
        interval = card.due - prev_due

        assert interval.days == 12

        # seed 2
        random.seed(12345)

        card = Card()
        card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=datetime.now(timezone.utc)
        )
        card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=card.due
        )
        prev_due = card.due
        card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=card.due
        )
        interval = card.due - prev_due

        assert interval.days == 11

    def test_no_learning_steps(self):
        scheduler = Scheduler(learning_steps=())

        assert len(scheduler.learning_steps) == 0

        card = Card()
        card, _ = scheduler.review_card(
            card=card, rating=Rating.Again, review_datetime=datetime.now(timezone.utc)
        )

        assert card.state == State.Review
        interval = (card.due - card.last_review).days
        assert interval >= 1

    def test_no_relearning_steps(self):
        scheduler = Scheduler(relearning_steps=())

        assert len(scheduler.relearning_steps) == 0

        card = Card()
        card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=datetime.now(timezone.utc)
        )
        assert card.state == State.Learning
        card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=card.due
        )
        assert card.state == State.Review
        card, _ = scheduler.review_card(
            card=card, rating=Rating.Again, review_datetime=card.due
        )
        assert card.state == State.Review

        interval = (card.due - card.last_review).days
        assert interval >= 1

    def test_one_card_multiple_schedulers(self):
        scheduler_with_two_learning_steps = Scheduler(
            learning_steps=(timedelta(minutes=1), timedelta(minutes=10))
        )
        scheduler_with_one_learning_step = Scheduler(
            learning_steps=(timedelta(minutes=1),)
        )
        scheduler_with_no_learning_steps = Scheduler(learning_steps=())

        scheduler_with_two_relearning_steps = Scheduler(
            relearning_steps=(
                timedelta(minutes=1),
                timedelta(minutes=10),
            )
        )
        scheduler_with_one_relearning_step = Scheduler(
            relearning_steps=(timedelta(minutes=1),)
        )
        scheduler_with_no_relearning_steps = Scheduler(relearning_steps=())

        card = Card()

        # learning-state tests
        assert len(scheduler_with_two_learning_steps.learning_steps) == 2
        card, _ = scheduler_with_two_learning_steps.review_card(
            card=card, rating=Rating.Good, review_datetime=datetime.now(timezone.utc)
        )
        assert card.state == State.Learning
        assert card.step == 1

        assert len(scheduler_with_one_learning_step.learning_steps) == 1
        card, _ = scheduler_with_one_learning_step.review_card(
            card=card, rating=Rating.Again, review_datetime=datetime.now(timezone.utc)
        )
        assert card.state == State.Learning
        assert card.step == 0

        assert len(scheduler_with_no_learning_steps.learning_steps) == 0
        card, _ = scheduler_with_no_learning_steps.review_card(
            card=card, rating=Rating.Hard, review_datetime=datetime.now(timezone.utc)
        )
        assert card.state == State.Review
        assert card.step is None

        # relearning-state tests
        assert len(scheduler_with_two_relearning_steps.relearning_steps) == 2
        card, _ = scheduler_with_two_relearning_steps.review_card(
            card=card, rating=Rating.Again, review_datetime=datetime.now(timezone.utc)
        )
        assert card.state == State.Relearning
        assert card.step == 0

        card, _ = scheduler_with_two_relearning_steps.review_card(
            card=card, rating=Rating.Good, review_datetime=datetime.now(timezone.utc)
        )
        assert card.state == State.Relearning
        assert card.step == 1

        assert len(scheduler_with_one_relearning_step.relearning_steps) == 1
        card, _ = scheduler_with_one_relearning_step.review_card(
            card=card, rating=Rating.Again, review_datetime=datetime.now(timezone.utc)
        )
        assert card.state == State.Relearning
        assert card.step == 0

        assert len(scheduler_with_no_relearning_steps.relearning_steps) == 0
        card, _ = scheduler_with_no_relearning_steps.review_card(
            card=card, rating=Rating.Hard, review_datetime=datetime.now(timezone.utc)
        )
        assert card.state == State.Review
        assert card.step is None

    def test_maximum_interval(self):
        maximum_interval = 100
        scheduler = Scheduler(maximum_interval=maximum_interval)

        card = Card()

        card, _ = scheduler.review_card(
            card=card, rating=Rating.Easy, review_datetime=card.due
        )
        assert (card.due - card.last_review).days <= scheduler.maximum_interval

        card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=card.due
        )
        assert (card.due - card.last_review).days <= scheduler.maximum_interval

        card, _ = scheduler.review_card(
            card=card, rating=Rating.Easy, review_datetime=card.due
        )
        assert (card.due - card.last_review).days <= scheduler.maximum_interval

        card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=card.due
        )
        assert (card.due - card.last_review).days <= scheduler.maximum_interval

    def test_class_repr(self):
        card = Card()

        assert str(card) == repr(card)

        scheduler = Scheduler()

        assert str(scheduler) == repr(scheduler)

        card, review_log = scheduler.review_card(card=card, rating=Rating.Good)

        assert str(review_log) == repr(review_log)

    def test_unique_card_ids(self):
        card_ids = []
        for i in range(1000):
            card = Card()
            card_id = card.card_id
            card_ids.append(card_id)

        assert len(card_ids) == len(set(card_ids))

    def test_stability_lower_bound(self):
        """
        Ensure that a Card object's stability is always >= STABILITY_MIN
        """

        scheduler = Scheduler()

        card = Card()

        for _ in range(1000):
            card, _ = scheduler.review_card(
                card=card,
                rating=Rating.Again,
                review_datetime=card.due + timedelta(days=1),
            )
            assert card.stability >= STABILITY_MIN

    def test_scheduler_parameter_validation(self):
        # initializing a Scheduler object with valid parameters works
        good_parameters = DEFAULT_PARAMETERS
        assert type(Scheduler(parameters=good_parameters)) is Scheduler

        parameters_one_too_high = list(DEFAULT_PARAMETERS)
        parameters_one_too_high[6] = 100
        with pytest.raises(ValueError):
            Scheduler(parameters=parameters_one_too_high)

        parameters_one_too_low = list(DEFAULT_PARAMETERS)
        parameters_one_too_low[10] = -42
        with pytest.raises(ValueError):
            Scheduler(parameters=parameters_one_too_low)

        parameters_two_bad = list(DEFAULT_PARAMETERS)
        parameters_two_bad[0] = 0
        parameters_two_bad[3] = 101
        with pytest.raises(ValueError):
            Scheduler(parameters=parameters_two_bad)

        zero_parameters = []
        with pytest.raises(ValueError):
            Scheduler(parameters=zero_parameters)

        one_too_few_parameters = DEFAULT_PARAMETERS[:-1]
        with pytest.raises(ValueError):
            Scheduler(parameters=one_too_few_parameters)

        too_many_parameters = DEFAULT_PARAMETERS + (1, 2, 3)
        with pytest.raises(ValueError):
            Scheduler(parameters=too_many_parameters)

    def test_class___eq___methods(self):
        scheduler1 = Scheduler()
        scheduler2 = Scheduler(desired_retention=0.91)
        scheduler1_copy = deepcopy(scheduler1)

        assert scheduler1 != scheduler2
        assert scheduler1 == scheduler1_copy

        card_orig = Card()
        card_orig_copy = deepcopy(card_orig)

        assert card_orig == card_orig_copy

        card_review_1, review_log_review_1 = scheduler1.review_card(
            card=card_orig, rating=Rating.Good
        )

        review_log_review_1_copy = deepcopy(review_log_review_1)

        assert card_orig != card_review_1
        assert review_log_review_1 == review_log_review_1_copy

        _, review_log_review_2 = scheduler1.review_card(
            card=card_review_1, rating=Rating.Good
        )

        assert review_log_review_1 != review_log_review_2

    def test_learning_card_rate_hard_one_learning_step(self):
        first_learning_step = timedelta(minutes=10)

        scheduler_with_one_learning_step = Scheduler(
            learning_steps=(first_learning_step,)
        )

        card = Card()

        initial_due_datetime = card.due

        card, _ = scheduler_with_one_learning_step.review_card(
            card=card, rating=Rating.Hard, review_datetime=card.due
        )

        assert card.state == State.Learning

        new_due_datetime = card.due

        interval_length = new_due_datetime - initial_due_datetime

        expected_interval_length = first_learning_step * 1.5

        tolerance = timedelta(seconds=1)

        assert abs(interval_length - expected_interval_length) <= tolerance

    def test_learning_card_rate_hard_second_learning_step(self):
        first_learning_step = timedelta(minutes=1)
        second_learning_step = timedelta(minutes=10)

        scheduler_with_two_learning_steps = Scheduler(
            learning_steps=(first_learning_step, second_learning_step)
        )

        card = Card()

        assert card.state == State.Learning
        assert card.step == 0

        card, _ = scheduler_with_two_learning_steps.review_card(
            card=card, rating=Rating.Good, review_datetime=card.due
        )

        assert card.state == State.Learning
        assert card.step == 1

        due_datetime_after_first_review = card.due

        card, _ = scheduler_with_two_learning_steps.review_card(
            card=card,
            rating=Rating.Hard,
            review_datetime=due_datetime_after_first_review,
        )

        due_datetime_after_second_review = card.due

        assert card.state == State.Learning
        assert card.step == 1

        interval_length = (
            due_datetime_after_second_review - due_datetime_after_first_review
        )

        expected_interval_length = second_learning_step

        tolerance = timedelta(seconds=1)

        assert abs(interval_length - expected_interval_length) <= tolerance

    def test_long_term_stability_learning_state(self):
        # NOTE: currently, this test is mostly to make sure that
        # the unit tests cover the case when a card in the relearning state
        # is not reviewed on the same day to run the non-same-day stability calculations

        scheduler = Scheduler()

        card = Card()

        assert card.state == State.Learning

        card, _ = scheduler.review_card(
            card=card, rating=Rating.Easy, review_datetime=card.due
        )

        assert card.state == State.Review

        card, _ = scheduler.review_card(
            card=card, rating=Rating.Again, review_datetime=card.due
        )

        assert card.state == State.Relearning

        relearning_card_due_datetime = card.due

        # a full day after its next due date
        next_review_datetime_one_day_late = relearning_card_due_datetime + timedelta(
            days=1
        )

        card, _ = scheduler.review_card(
            card=card,
            rating=Rating.Good,
            review_datetime=next_review_datetime_one_day_late,
        )

        assert card.state == State.Review

    def test_relearning_card_rate_hard_one_relearning_step(self):
        first_relearning_step = timedelta(minutes=10)

        scheduler_with_one_relearning_step = Scheduler(
            relearning_steps=(first_relearning_step,)
        )

        card = Card()

        card, _ = scheduler_with_one_relearning_step.review_card(
            card=card, rating=Rating.Easy, review_datetime=card.due
        )

        assert card.state == State.Review

        card, _ = scheduler_with_one_relearning_step.review_card(
            card=card, rating=Rating.Again, review_datetime=card.due
        )

        assert card.state == State.Relearning
        assert card.step == 0

        prev_due_datetime = card.due

        card, _ = scheduler_with_one_relearning_step.review_card(
            card=card, rating=Rating.Hard, review_datetime=prev_due_datetime
        )

        assert card.state == State.Relearning
        assert card.step == 0

        new_due_datetime = card.due

        interval_length = new_due_datetime - prev_due_datetime

        expected_interval_length = first_relearning_step * 1.5

        tolerance = timedelta(seconds=1)

        assert abs(interval_length - expected_interval_length) <= tolerance

    def test_relearning_card_rate_hard_two_relearning_steps(self):
        first_relearning_step = timedelta(minutes=1)
        second_relearning_step = timedelta(minutes=10)

        scheduler_with_two_relearning_steps = Scheduler(
            relearning_steps=(first_relearning_step, second_relearning_step)
        )

        card = Card()

        card, _ = scheduler_with_two_relearning_steps.review_card(
            card=card, rating=Rating.Easy, review_datetime=card.due
        )

        assert card.state == State.Review

        card, _ = scheduler_with_two_relearning_steps.review_card(
            card=card, rating=Rating.Again, review_datetime=card.due
        )

        assert card.state == State.Relearning
        assert card.step == 0

        prev_due_datetime = card.due

        card, _ = scheduler_with_two_relearning_steps.review_card(
            card=card, rating=Rating.Hard, review_datetime=prev_due_datetime
        )

        assert card.state == State.Relearning
        assert card.step == 0

        new_due_datetime = card.due

        interval_length = new_due_datetime - prev_due_datetime

        expected_interval_length = (
            first_relearning_step + second_relearning_step
        ) / 2.0

        tolerance = timedelta(seconds=1)

        assert abs(interval_length - expected_interval_length) <= tolerance

        card, _ = scheduler_with_two_relearning_steps.review_card(
            card=card, rating=Rating.Good, review_datetime=card.due
        )

        assert card.state == State.Relearning
        assert card.step == 1

        prev_due_datetime = card.due

        card, _ = scheduler_with_two_relearning_steps.review_card(
            card=card, rating=Rating.Hard, review_datetime=prev_due_datetime
        )

        new_due_datetime = card.due

        assert card.state == State.Relearning
        assert card.step == 1

        interval_length = new_due_datetime - prev_due_datetime

        expected_interval_length = second_relearning_step

        tolerance = timedelta(seconds=1)

        assert abs(interval_length - expected_interval_length) <= tolerance

        card, _ = scheduler_with_two_relearning_steps.review_card(
            card=card, rating=Rating.Easy, review_datetime=prev_due_datetime
        )

        assert card.state == State.Review
        assert card.step is None

    def test_reschedule_card_same_scheduler(self):
        scheduler = Scheduler(enable_fuzzing=False)

        card = Card()

        review_logs = []
        for rating in TEST_RATINGS_1:
            card, review_log = scheduler.review_card(
                card=card, rating=rating, review_datetime=card.due
            )
            review_logs.append(review_log)

        rescheduled_card = scheduler.reschedule_card(card=card, review_logs=review_logs)

        assert card is not rescheduled_card
        assert card == rescheduled_card

    def test_reschedule_card_different_parameters(self):
        scheduler = Scheduler(enable_fuzzing=False)

        card = Card()

        review_logs = []
        for rating in TEST_RATINGS_1:
            card, review_log = scheduler.review_card(
                card=card, rating=rating, review_datetime=card.due
            )
            review_logs.append(review_log)

        DIFFERENT_PARAMETERS = [
            0.12340357383516173,
            1.2931,
            2.397673571899466,
            8.2956,
            6.686820427099132,
            0.45021679958387956,
            3.077875127553957,
            0.053520395733247045,
            1.6539992229052127,
            0.1466206769107436,
            0.6300772488850335,
            1.611965002575047,
            0.012840136810798864,
            0.34853762746216305,
            1.8878958285806287,
            0.8546376191171063,
            1.8729,
            0.6748536823468675,
            0.20451266082721842,
            0.22622814695113844,
            0.46030603398979064,
        ]
        assert scheduler.parameters != DIFFERENT_PARAMETERS
        scheduler_with_different_parameters = Scheduler(parameters=DIFFERENT_PARAMETERS)
        rescheduled_card = scheduler_with_different_parameters.reschedule_card(
            card=card, review_logs=review_logs
        )

        assert card.card_id == rescheduled_card.card_id
        assert card.state == rescheduled_card.state
        assert card.step == rescheduled_card.step
        assert card.stability != rescheduled_card.stability
        assert card.difficulty != rescheduled_card.difficulty
        assert card.last_review == rescheduled_card.last_review
        assert card.due != rescheduled_card.due

    def test_reschedule_card_different_desired_retention(self):
        scheduler = Scheduler(enable_fuzzing=False)

        card = Card()

        review_logs = []
        for rating in TEST_RATINGS_1:
            card, review_log = scheduler.review_card(
                card=card, rating=rating, review_datetime=card.due
            )
            review_logs.append(review_log)

        DIFFERENT_DESIRED_RETENTION = 0.8
        assert scheduler.desired_retention != DIFFERENT_DESIRED_RETENTION
        scheduler_with_different_retention = Scheduler(
            desired_retention=DIFFERENT_DESIRED_RETENTION
        )
        rescheduled_card = scheduler_with_different_retention.reschedule_card(
            card=card, review_logs=review_logs
        )

        assert card.card_id == rescheduled_card.card_id
        assert card.state == rescheduled_card.state
        assert card.step == rescheduled_card.step
        assert card.stability == rescheduled_card.stability
        assert card.difficulty == rescheduled_card.difficulty
        assert card.last_review == rescheduled_card.last_review
        assert card.due < rescheduled_card.due

    def test_reschedule_card_different_learning_steps(self):
        scheduler = Scheduler(enable_fuzzing=False)

        card = Card()

        review_logs = []
        for rating in TEST_RATINGS_1:
            card, review_log = scheduler.review_card(
                card=card, rating=rating, review_datetime=card.due
            )
            review_logs.append(review_log)

        DIFFERENT_LEARNING_STEPS = [timedelta(minutes=1)] * len(review_logs)
        assert scheduler.learning_steps != DIFFERENT_LEARNING_STEPS
        scheduler_with_different_retention = Scheduler(
            learning_steps=DIFFERENT_LEARNING_STEPS
        )
        rescheduled_card = scheduler_with_different_retention.reschedule_card(
            card=card, review_logs=review_logs
        )

        assert card.card_id == rescheduled_card.card_id
        assert card.state != rescheduled_card.state
        assert card.step != rescheduled_card.step
        assert card.stability == rescheduled_card.stability
        assert card.difficulty == rescheduled_card.difficulty
        assert card.last_review == rescheduled_card.last_review
        assert card.due > rescheduled_card.due

    def test_reschedule_card_wrong_review_logs(self):
        scheduler = Scheduler(enable_fuzzing=False)

        card = Card()

        review_logs = []
        for rating in TEST_RATINGS_1:
            card, review_log = scheduler.review_card(
                card=card, rating=rating, review_datetime=card.due
            )
            review_logs.append(review_log)

        DIFFERENT_CARD_ID = 123
        assert card.card_id != DIFFERENT_CARD_ID
        review_logs[0].card_id = DIFFERENT_CARD_ID

        EXPECTED_ERROR_MESSAGE = f"ReviewLog card_id {DIFFERENT_CARD_ID} does not match Card card_id {card.card_id}"
        with pytest.raises(ValueError, match=re.escape(EXPECTED_ERROR_MESSAGE)):
            _ = scheduler.reschedule_card(card=card, review_logs=review_logs)

    def test_Optimizer_lazy_loading(self):
        assert "fsrs.scheduler" in sys.modules
        assert "fsrs.card" in sys.modules
        assert "fsrs.review_log" in sys.modules

        assert "fsrs.optimizer" not in sys.modules

        from fsrs import Optimizer  # noqa: F401 (linter: unused import)

        assert "fsrs.optimizer" in sys.modules

    def test_import_non_existent_module(self):
        with pytest.raises(ImportError):
            from fsrs import NotAModule  # noqa: F401 (linter: unused import)
