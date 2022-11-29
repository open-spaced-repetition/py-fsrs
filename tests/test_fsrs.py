from fsrs import FSRS, Card
from datetime import datetime
import json


def print_scheduling_cards(scheduling_cards):
    print("again:", json.dumps(scheduling_cards.again.__dict__))
    print(scheduling_cards.record_log()['again'].__dict__)
    print("hard:", json.dumps(scheduling_cards.hard.__dict__))
    print(scheduling_cards.record_log()['hard'].__dict__)
    print("good:", json.dumps(scheduling_cards.good.__dict__))
    print(scheduling_cards.record_log()['good'].__dict__)
    print("easy:", json.dumps(scheduling_cards.easy.__dict__))
    print(scheduling_cards.record_log()['easy'].__dict__)
    print()


def test_repeat():
    fsrs = FSRS()
    card = Card()
    now = int(datetime(2000, 7, 30, 12).timestamp() * 1000)
    scheduling_cards = fsrs.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards.good
    now = card.due
    scheduling_cards = fsrs.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards.good
    now = card.due
    scheduling_cards = fsrs.repeat(card, now)
    print_scheduling_cards(scheduling_cards)

    card = scheduling_cards.good
    now = card.due
    scheduling_cards = fsrs.repeat(card, now)
    print_scheduling_cards(scheduling_cards)


test_repeat()
