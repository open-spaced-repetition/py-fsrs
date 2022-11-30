from fsrs import *
from datetime import datetime
import json


def print_scheduling_cards(scheduling_cards):
    print("again.card:", scheduling_cards[AGAIN].card.__dict__)
    print("again.review_log:", scheduling_cards[AGAIN].review_log.__dict__)
    print("hard.card:", scheduling_cards[HARD].card.__dict__)
    print("hard.review_log:", scheduling_cards[HARD].review_log.__dict__)
    print("good.card:", scheduling_cards[GOOD].card.__dict__)
    print("good.review_log:", scheduling_cards[GOOD].review_log.__dict__)
    print("easy.card:", scheduling_cards[EASY].card.__dict__)
    print("easy.review_log:", scheduling_cards[EASY].review_log.__dict__)
    print()


def test_repeat():
    f = FSRS()
    card = Card()
    now = datetime(2022, 11, 29, 12, 30, 0, 0)
    scheduling_cards = f.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards[GOOD].card
    now = card.due
    scheduling_cards = f.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards[GOOD].card
    now = card.due
    scheduling_cards = f.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards[AGAIN].card
    now = card.due
    scheduling_cards = f.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards[GOOD].card
    now = card.due
    scheduling_cards = f.repeat(card, now)
    print_scheduling_cards(scheduling_cards)


test_repeat()
