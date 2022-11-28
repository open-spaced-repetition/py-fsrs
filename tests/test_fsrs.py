from fsrs import FSRS, Card
from datetime import datetime

def todict(obj):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v)
        return data
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif hasattr(obj, "__iter__"):
        return [todict(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, todict(value))
                     for key, value in obj.__dict__.items()
                     if not callable(value) and not key.startswith('_') and key not in ['name']])
        if hasattr(obj, "name") and hasattr(obj, "value") :
            data[obj.name] = obj.value
        return data
    else:
        return obj

def test_repeat():
    fsrs = FSRS()
    card = Card()
    now = datetime(2000, 7, 30, 12)
    scheduling_cards = fsrs.repeat(card, now)
    print(todict(scheduling_cards))

    card = scheduling_cards.good
    now = card.due
    scheduling_cards = fsrs.repeat(card, now)
    print(todict(scheduling_cards))

    card = scheduling_cards.again
    now = card.due
    scheduling_cards = fsrs.repeat(card, now)
    print(todict(scheduling_cards))

    card = scheduling_cards.good
    now = card.due
    scheduling_cards = fsrs.repeat(card, now)
    print(todict(scheduling_cards))

test_repeat()