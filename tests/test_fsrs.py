from fsrs import *
from datetime import datetime


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


def test_repeat():
    f = FSRS()
    card = Card()
    now = datetime(2022, 11, 29, 12, 30, 0, 0)
    scheduling_cards = f.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards[Rating.Good].card
    now = card.due
    scheduling_cards = f.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards[Rating.Good].card
    now = card.due
    scheduling_cards = f.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards[Rating.Again].card
    now = card.due
    scheduling_cards = f.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards[Rating.Good].card
    now = card.due
    scheduling_cards = f.repeat(card, now)
    print_scheduling_cards(scheduling_cards)


test_repeat()
