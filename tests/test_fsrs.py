from fsrs import *
from datetime import datetime, timezone
import json
import pytest


def print_scheduling_cards(scheduling_cards):
    print("again.card:", scheduling_cards[Rating.Again].card.__dict__)
    print("again.review_log:", scheduling_cards[Rating.Again].review_log.__dict__)
    print("hard.card:", scheduling_cards[Rating.Hard].card.__dict__)
    print("hard.review_log:", scheduling_cards[Rating.Hard].review_log.__dict__)
    print("good.card:", scheduling_cards[Rating.Good].card.__dict__)
    print("good.review_log:", scheduling_cards[Rating.Good].review_log.__dict__)
    print("easy.card:", scheduling_cards[Rating.Easy].card.__dict__)
    print("easy.review_log:", scheduling_cards[Rating.Easy].review_log.__dict__)
    print()


class TestPyFSRS:

    def test_repeat(self):
        f = FSRS()
        f.p.w = (
            1.14,
            1.01,
            5.44,
            14.67,
            5.3024,
            1.5662,
            1.2503,
            0.0028,
            1.5489,
            0.1763,
            0.9953,
            2.7473,
            0.0179,
            0.3105,
            0.3976,
            0.0,
            2.0902,
        )
        card = Card()
        now = datetime(2022, 11, 29, 12, 30, 0, 0, timezone.utc)
        scheduling_cards = f.repeat(card, now)
        print_scheduling_cards(scheduling_cards)

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
            card = scheduling_cards[rating].card
            ivl = card.scheduled_days
            ivl_history.append(ivl)
            now = card.due
            scheduling_cards = f.repeat(card, now)
            print_scheduling_cards(scheduling_cards)

        print(ivl_history)
        assert ivl_history == [0, 5, 16, 43, 106, 236, 0, 0, 12, 25, 47, 85, 147]

    def test_repeat_default_arg(self):

        f = FSRS()

        card_object = Card()

        # repeat time is not specified
        scheduling_cards = f.repeat(card_object)

        card_rating = Rating.Good

        card_object = scheduling_cards[card_rating].card

        due = card_object.due

        time_delta = due - datetime.now(timezone.utc)

        assert time_delta.seconds > 500  # due in approx. 8-10 minutes

    def test_datetime(self):

        f = FSRS()
        card = Card()

        # new cards should be due immediately after creation
        assert datetime.now(timezone.utc) >= card.due

        # comparing timezone aware cards with deprecated datetime.utcnow() should raise a TypeError
        with pytest.raises(TypeError):
            datetime.now() >= card.due

        # repeating a card with a non-utc, non-timezone-aware datetime object should raise a Value Error
        with pytest.raises(ValueError):
            f.repeat(card, datetime(2022, 11, 29, 12, 30, 0, 0))

        # repeat a card with rating good before next tests
        scheduling_cards = f.repeat(card, datetime.now(timezone.utc))
        card = scheduling_cards[Rating.Good].card

        # card object's due and last_review attributes must be timezone aware and UTC
        assert card.due.tzinfo == timezone.utc
        assert card.last_review.tzinfo == timezone.utc
        # card object's due datetime should be later than its last review
        assert card.due >= card.last_review

    def test_serialize(self):

        f = FSRS()

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
        scheduling_cards = f.repeat(card, datetime.now(timezone.utc))
        repeated_card = scheduling_cards[Rating.Good].card

        with pytest.raises(TypeError):
            json.dumps(repeated_card.__dict__)

        assert type(json.dumps(repeated_card.to_dict())) == str

        repeated_card_dict = repeated_card.to_dict()
        copied_repeated_card = Card.from_dict(repeated_card_dict)

        assert vars(repeated_card) == vars(copied_repeated_card)
        assert repeated_card.to_dict() == copied_repeated_card.to_dict()

        # original card and repeated card are different
        assert vars(card) != vars(repeated_card)
        assert card.to_dict() != repeated_card.to_dict()
