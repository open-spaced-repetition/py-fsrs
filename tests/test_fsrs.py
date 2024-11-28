from fsrs import FSRSScheduler, Card, ReviewLog, State, Rating
from datetime import datetime, timedelta, timezone
import json
import pytest

test_parameters = (
    0.4197,
    1.1869,
    3.0412,
    15.2441,
    7.1434,
    0.6477,
    1.0007,
    0.0674,
    1.6597,
    0.1712,
    1.1178,
    2.0225,
    0.0904,
    0.3025,
    2.1214,
    0.2498,
    2.9466,
    0.4891,
    0.6468,
)


class TestPyFSRS:
    def test_review_card(self):

        scheduler = FSRSScheduler(parameters=test_parameters, enable_fuzzing=False)

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
            card, _ = scheduler.review_card(card=card, rating=rating, review_datetime=review_datetime)

            ivl = (card.due - card.last_review).days
            ivl_history.append(ivl)

            review_datetime = card.due

        assert ivl_history == [
            0,
            4,
            17,
            62,
            198,
            563,
            0,
            0,
            9,
            27,
            74,
            190,
            457,
        ]

    def test_memo_state(self):
    
        scheduler = FSRSScheduler(parameters=test_parameters)

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

            card, _ = scheduler.review_card(card=card, rating=rating, review_datetime=review_datetime)

            review_datetime += timedelta(days=ivl)

        card, _ = scheduler.review_card(card=card, rating=Rating.Good, review_datetime=review_datetime)
            
        assert round(card.stability, 4) == 71.4554
        assert round(card.difficulty, 4) == 5.0976

    def test_repeat_default_arg(self):
    
        scheduler = FSRSScheduler()

        card = Card()

        rating = Rating.Good

        card, _ = scheduler.review_card(card=card, rating=rating,)

        due = card.due

        time_delta = due - datetime.now(timezone.utc)

        assert time_delta.seconds > 500  # due in approx. 8-10 minutes

    def test_datetime(self):
    
        scheduler = FSRSScheduler()
        card = Card()

        # new cards should be due immediately after creation
        assert datetime.now(timezone.utc) >= card.due

        # comparing timezone aware cards with deprecated datetime.utcnow() should raise a TypeError
        with pytest.raises(TypeError):
            datetime.now() >= card.due

        # repeating a card with a non-utc, non-timezone-aware datetime object should raise a Value Error
        with pytest.raises(ValueError):
            scheduler.review_card(card=card, rating=Rating.Good, review_datetime=datetime(2022, 11, 29, 12, 30, 0, 0))

        # review a card with rating good before next tests
        card, _ = scheduler.review_card(card=card, rating=Rating.Good, review_datetime=datetime.now(timezone.utc))

        # card object's due and last_review attributes must be timezone aware and UTC
        assert card.due.tzinfo == timezone.utc
        assert card.last_review.tzinfo == timezone.utc
        # card object's due datetime should be later than its last review
        assert card.due >= card.last_review

    def test_Card_serialize(self):
    
        scheduler = FSRSScheduler()

        # create card object the normal way
        card = Card()

        # card object is not naturally JSON serializable
        with pytest.raises(TypeError):
            json.dumps(card.__dict__)

        # card object's to_dict() method makes it JSON serializable
        assert type(json.dumps(card.to_dict())) == str

        # we can reconstruct a copy of the card object equivalent to the original
        card_dict = card.to_dict()
        copied_card = Card.from_dict(card_dict)

        assert vars(card) == vars(copied_card)
        assert card.to_dict() == copied_card.to_dict()

        # (x2) perform the above tests once more with a repeated card
        reviewed_card, _ = scheduler.review_card(card=card, rating=Rating.Good, review_datetime=datetime.now(timezone.utc))

        with pytest.raises(TypeError):
            json.dumps(reviewed_card.__dict__)

        assert type(json.dumps(reviewed_card.to_dict())) == str

        reviewed_card_dict = reviewed_card.to_dict()
        copied_reviewed_card = Card.from_dict(reviewed_card_dict)

        assert vars(reviewed_card) == vars(copied_reviewed_card)
        assert reviewed_card.to_dict() == copied_reviewed_card.to_dict()

        # original card and repeated card are different
        assert vars(card) != vars(reviewed_card)
        assert card.to_dict() != reviewed_card.to_dict()

    def test_ReviewLog_serialize(self):
    
        scheduler = FSRSScheduler()

        card = Card()

        # review a card to get the review_log
        card, review_log = scheduler.review_card(card=card, rating=Rating.Again)

        # ReviewLog object is not naturally JSON serializable
        with pytest.raises(TypeError):
            json.dumps(review_log.__dict__)

        # review_log object's to_dict() method makes it JSON serializable
        assert type(json.dumps(review_log.to_dict())) == str

        # we can reconstruct a copy of the review_log object equivalent to the original
        review_log_dict = review_log.to_dict()
        copied_review_log = ReviewLog.from_dict(review_log_dict)
        assert review_log.to_dict() == copied_review_log.to_dict()

        # (x2) perform the above tests once more with a review_log from a reviewed card
        rating = Rating.Good
        card, next_review_log = scheduler.review_card(card=card, rating=rating, review_datetime=datetime.now(timezone.utc))

        with pytest.raises(TypeError):
            json.dumps(next_review_log.__dict__)

        assert type(json.dumps(next_review_log.to_dict())) == str

        next_review_log_dict = next_review_log.to_dict()
        copied_next_review_log = ReviewLog.from_dict(next_review_log_dict)

        assert next_review_log.to_dict() == copied_next_review_log.to_dict()

        # original review log and next review log are different
        assert review_log.to_dict() != next_review_log.to_dict()

    def test_custom_scheduler_args(self):
    
        scheduler = FSRSScheduler(
            parameters=(
                0.4197,
                1.1869,
                3.0412,
                15.2441,
                7.1434,
                0.6477,
                1.0007,
                0.0674,
                1.6597,
                0.1712,
                1.1178,
                2.0225,
                0.0904,
                0.3025,
                2.1214,
                0.2498,
                2.9466,
                0,
                0.6468,
            ),
            desired_retention=0.9,
            maximum_interval=36500,
            enable_fuzzing=False
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

        assert ivl_history == [0, 3, 13, 50, 163, 473, 0, 0, 12, 34, 91, 229, 541]

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
        scheduler2 = FSRSScheduler(
            parameters=parameters2,
            desired_retention=desired_retention2,
            maximum_interval=maximum_interval2,
        )

        assert scheduler2.parameters == parameters2
        assert scheduler2.desired_retention == desired_retention2
        assert scheduler2.maximum_interval == maximum_interval2

    def test_retrievability(self):
    
        scheduler = FSRSScheduler()

        card = Card()

        # retrievabiliy of New card
        assert card.state == State.New
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

        scheduler = FSRSScheduler()

        # FSRSScheduler objects are json-serializable through its .to_dict() method
        assert type(json.dumps(scheduler.to_dict())) == str

        # scheduler can be serialized and de-serialized while remaining the same
        scheduler_dict = scheduler.to_dict()
        copied_scheduler = FSRSScheduler.from_dict(scheduler_dict)
        assert vars(scheduler) == vars(copied_scheduler)
        assert scheduler.to_dict() == copied_scheduler.to_dict()

    def test_basic_fuzzed_interval(self):

        scheduler = FSRSScheduler(parameters=test_parameters, enable_fuzzing=True)

        ratings = (
            Rating.Good,
            Rating.Good,
            Rating.Good
        )
        
        card = Card()
        review_datetime = datetime(2022, 11, 29, 12, 30, 0, 0, timezone.utc)

        ivl_history = []
        for rating in ratings:
            card, _ = scheduler.review_card(card=card, rating=rating, review_datetime=review_datetime)

            ivl = (card.due - card.last_review).days
            ivl_history.append(ivl)

            review_datetime = card.due

        assert ivl_history[0] == 0 # review New-state card (no fuzz)
        assert ivl_history[1] == 4 # review Learning-state card (no fuzz)
        assert 14 <= ivl_history[2] <= 21 # review Review-state card (fuzz)