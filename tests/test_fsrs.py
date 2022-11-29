from fsrs import *
from datetime import datetime
import json


def print_scheduling_cards(scheduling_cards):
    print("again.card:", json.dumps(scheduling_cards[AGAIN].card.__dict__))
    print("again.review_log:", json.dumps(scheduling_cards[AGAIN].review_log.__dict__))
    print("hard.card:", json.dumps(scheduling_cards[HARD].card.__dict__))
    print("hard.review_log:", json.dumps(scheduling_cards[HARD].review_log.__dict__))
    print("good.card:", json.dumps(scheduling_cards[GOOD].card.__dict__))
    print("good.review_log:", json.dumps(scheduling_cards[GOOD].review_log.__dict__))
    print("easy.card:", json.dumps(scheduling_cards[EASY].card.__dict__))
    print("easy.review_log:", json.dumps(scheduling_cards[EASY].review_log.__dict__))
    print()


def test_repeat():
    fsrs = FSRS()
    card = Card()
    now = int(datetime(2000, 7, 30, 12).timestamp() * 1000)
    scheduling_cards = fsrs.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards[GOOD].card
    now = card.due
    scheduling_cards = fsrs.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards[GOOD].card
    now = card.due
    scheduling_cards = fsrs.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards[GOOD].card
    now = card.due
    scheduling_cards = fsrs.repeat(card, now)
    print_scheduling_cards(scheduling_cards)


test_repeat()
