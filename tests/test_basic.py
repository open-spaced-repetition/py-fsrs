from fsrs import Scheduler, Card, ReviewLog, State, Rating
from datetime import datetime, timedelta, timezone
import json
import pytest
import random


class TestPyFSRS:
    def test_review_card(self):
        scheduler = Scheduler(enable_fuzzing=False)

        ratings = (
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

        card = Card()
        review_datetime = datetime(2022, 11, 29, 12, 30, 0, 0, timezone.utc)

        ivl_history = []
        for rating in ratings:
            card, _ = scheduler.review_card(
                card=card, rating=rating, review_datetime=review_datetime
            )

            ivl = (card.due - card.last_review).days
            ivl_history.append(ivl)

            review_datetime = card.due

        assert ivl_history == [
            0,
            4,
            14,
            44,
            125,
            328,
            0,
            0,
            7,
            16,
            34,
            71,
            142,
        ]

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
            card, _ = scheduler.review_card(
                card=card, rating=rating, review_datetime=review_datetime
            )

            review_datetime += timedelta(days=ivl)

        card, _ = scheduler.review_card(
            card=card, rating=Rating.Good, review_datetime=review_datetime
        )

        assert round(card.stability) == 49
        assert round(card.difficulty, 4) == 7.0866

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

    def test_Card_serialize(self):
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

    def test_ReviewLog_serialize(self):
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

    def test_custom_scheduler_args(self):
        scheduler = Scheduler(
            desired_retention=0.9,
            maximum_interval=36500,
            enable_fuzzing=False,
        )
        card = Card()
        now = datetime(2022, 11, 29, 12, 30, 0, 0, timezone.utc)

        ratings = (
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
        ivl_history = []

        for rating in ratings:
            card, _ = scheduler.review_card(card, rating, now)
            ivl = (card.due - card.last_review).days
            ivl_history.append(ivl)
            now = card.due

        assert ivl_history == [0, 4, 14, 44, 125, 328, 0, 0, 7, 16, 34, 71, 142]

        # initialize another scheduler and verify parameters are properly set
        parameters2 = (
            0.1456,
            0.4186,
            1.1104,
            4.1315,
            5.2417,
            1.3098,
            0.8975,
            0.0000,
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
            5.6789,
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
        retrievability = card.get_retrievability()
        assert retrievability == 0

        # retrievabiliy of Learning card
        card, _ = scheduler.review_card(card, Rating.Good)
        assert card.state == State.Learning
        retrievability = card.get_retrievability()
        assert 0 <= retrievability <= 1

        # retrievabiliy of Review card
        card, _ = scheduler.review_card(card, Rating.Good)
        assert card.state == State.Review
        retrievability = card.get_retrievability()
        assert 0 <= retrievability <= 1

        # retrievabiliy of Relearning card
        card, _ = scheduler.review_card(card, Rating.Again)
        assert card.state == State.Relearning
        retrievability = card.get_retrievability()
        assert 0 <= retrievability <= 1

    def test_Scheduler_serialize(self):
        scheduler = Scheduler()

        # Scheduler objects are json-serializable through its .to_dict() method
        assert type(json.dumps(scheduler.to_dict())) is str

        # scheduler can be serialized and de-serialized while remaining the same
        scheduler_dict = scheduler.to_dict()
        copied_scheduler = Scheduler.from_dict(scheduler_dict)
        assert vars(scheduler) == vars(copied_scheduler)
        assert scheduler.to_dict() == copied_scheduler.to_dict()

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

        assert interval.days == 13

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

        assert interval.days == 12

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
        scheduler_with_no_learning_steps = Scheduler(learning_steps=())

        card = Card()

        assert len(scheduler_with_two_learning_steps.learning_steps) == 2
        card, _ = scheduler_with_two_learning_steps.review_card(
            card=card, rating=Rating.Good, review_datetime=datetime.now(timezone.utc)
        )
        assert card.state == State.Learning
        assert card.step == 1

        assert len(scheduler_with_no_learning_steps.learning_steps) == 0
        card, _ = scheduler_with_no_learning_steps.review_card(
            card=card, rating=Rating.Again, review_datetime=datetime.now(timezone.utc)
        )
        assert card.state == State.Review
        assert card.step is None

        scheduler_with_two_relearning_steps = Scheduler(
            relearning_steps=(
                timedelta(minutes=1),
                timedelta(minutes=10),
                timedelta(minutes=15),
            )
        )
        scheduler_with_no_relearning_steps = Scheduler(relearning_steps=())

        assert len(scheduler_with_two_relearning_steps.relearning_steps) == 3
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

        card, _ = scheduler_with_two_relearning_steps.review_card(
            card=card, rating=Rating.Good, review_datetime=datetime.now(timezone.utc)
        )
        assert card.state == State.Relearning
        assert card.step == 2

        card, _ = scheduler_with_no_relearning_steps.review_card(
            card=card, rating=Rating.Again, review_datetime=datetime.now(timezone.utc)
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
